// Vulkan compute harness: device bring-up, quantized-matmul pipelines, weight
// residency. See vk-compute.h for the contract and model-vulkan.c for the
// caller. Deliberately one dispatch per submit with a fence wait — simple and
// correct first; the journal culture measures before batching.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <vulkan/vulkan.h>

#include "vk-compute.h"
#include "quant.h"

// SPIR-V for shaders/matmul.comp, one compilation per weight type (CMake runs
// glslc -DQTYPE=<id> -mfmt=num, which emits the words as C initializer text).
static const uint32_t spv_f32[] = {
#include "matmul_f32.spv.inc"
};
static const uint32_t spv_f16[] = {
#include "matmul_f16.spv.inc"
};
static const uint32_t spv_q4_0[] = {
#include "matmul_q4_0.spv.inc"
};
static const uint32_t spv_q8_0[] = {
#include "matmul_q8_0.spv.inc"
};
static const uint32_t spv_q4_K[] = {
#include "matmul_q4_K.spv.inc"
};
static const uint32_t spv_q6_K[] = {
#include "matmul_q6_K.spv.inc"
};
static const uint32_t spv_bf16[] = {
#include "matmul_bf16.spv.inc"
};

#define NTYPES 32              // ggml type ids fit (BF16 = 30 is the largest)
#define NSLOTS 4096            // weight-buffer cache (open addressing; ~400 tensors)
#define NWIN   16              // zero-copy windows: 16 x 1 GiB stride covers 17 GiB

struct wslot { const void *key; VkBuffer buf; VkDeviceMemory mem; uint32_t base; };

static struct {
    VkInstance            inst;
    VkPhysicalDevice      phys;
    VkDevice              dev;
    VkQueue               queue;
    uint32_t              qfam;
    VkCommandPool         pool;
    VkCommandBuffer       cb;
    VkFence               fence;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout      pl;
    VkDescriptorPool      dpool;
    VkDescriptorSet       ds;
    VkPipeline            pipe[NTYPES];
    VkPhysicalDeviceMemoryProperties mem;
    uint32_t              max_range;      // maxStorageBufferRange

    // activation I/O: persistently mapped, host-coherent, grown on demand
    VkBuffer       xbuf, obuf;
    VkDeviceMemory xmem, omem;
    void          *xmap, *omap;
    size_t         xcap, ocap;

    // zero-copy import of the GGUF blob (VK_EXT_external_memory_host)
    int            imported;
    VkDeviceMemory imp;
    const uint8_t *blob;
    size_t         blob_size, imp_size;
    size_t         win_stride;            // window w covers [w*stride, w*stride + win size)
    struct { VkBuffer buf; size_t size; } win[NWIN];
    int            nwin;

    // upload fallback: per-tensor buffers, cached by data pointer
    struct wslot   slots[NSLOTS];

    int ready;                            // 0 = untried, 1 = up, -1 = failed
} g;

// ---- small helpers ---------------------------------------------------------

static int find_mem_type(uint32_t type_bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < g.mem.memoryTypeCount; i++)
        if ((type_bits & (1u << i)) &&
            (g.mem.memoryTypes[i].propertyFlags & want) == want)
            return (int)i;
    return -1;
}

// What a mapped buffer is for decides which memory type it wants. GPU-read
// buffers (weights, x) must AVOID host-cached types: on an APU those keep CPU
// cache coherence and the GPU reads them through the snoop path at a fraction
// of DRAM speed — write-combined GTT reads full speed, and the CPU only ever
// memcpy's INTO these, which write-combining handles fine. The out buffer is
// the opposite: the GPU writes it once, the CPU reads it back, and CPU reads
// from write-combined memory are the slow thing — prefer host-cached there.
enum buf_kind { BUF_GPU_READ, BUF_CPU_READ };

static int pick_hostvis(uint32_t type_bits, enum buf_kind kind) {
    const VkMemoryPropertyFlags base = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (kind == BUF_CPU_READ) {
        int i = find_mem_type(type_bits, base | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (i >= 0) return i;
        return find_mem_type(type_bits, base);
    }
    // BUF_GPU_READ: device-local first, then anything host-visible that is
    // NOT host-cached, and host-cached only as the last resort.
    int best = -1, best_rank = -1;
    for (uint32_t i = 0; i < g.mem.memoryTypeCount; i++) {
        if (!(type_bits & (1u << i))) continue;
        VkMemoryPropertyFlags f = g.mem.memoryTypes[i].propertyFlags;
        if ((f & base) != base) continue;
        int rank = 1;
        if (!(f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) rank = 2;
        if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)   rank += 2;
        if (rank > best_rank) { best = (int)i; best_rank = rank; }
    }
    return best;
}

static int make_buffer(VkDeviceSize size, VkBuffer *buf, VkDeviceMemory *mem, void **map,
                       enum buf_kind kind) {
    VkBufferCreateInfo bi = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(g.dev, &bi, NULL, buf) != VK_SUCCESS) return -1;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g.dev, *buf, &req);
    int idx = pick_hostvis(req.memoryTypeBits, kind);
    if (idx < 0) { vkDestroyBuffer(g.dev, *buf, NULL); *buf = VK_NULL_HANDLE; return -1; }
    VkMemoryAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size, .memoryTypeIndex = (uint32_t)idx,
    };
    if (vkAllocateMemory(g.dev, &ai, NULL, mem) != VK_SUCCESS ||
        vkBindBufferMemory(g.dev, *buf, *mem, 0) != VK_SUCCESS) {
        vkDestroyBuffer(g.dev, *buf, NULL); *buf = VK_NULL_HANDLE;
        if (*mem) vkFreeMemory(g.dev, *mem, NULL);
        *mem = VK_NULL_HANDLE;
        return -1;
    }
    if (map && vkMapMemory(g.dev, *mem, 0, VK_WHOLE_SIZE, 0, map) != VK_SUCCESS) {
        vkDestroyBuffer(g.dev, *buf, NULL); vkFreeMemory(g.dev, *mem, NULL);
        *buf = VK_NULL_HANDLE; *mem = VK_NULL_HANDLE;
        return -1;
    }
    return 0;
}

// ---- zero-copy import of the GGUF blob -------------------------------------

// Bind ≤2 GiB storage-buffer windows every 1 GiB over the imported blob: any
// tensor ≤1 GiB then lies whole inside window (offset / stride), and the
// shader adds the tensor's byte offset within the window from a push constant
// — no descriptor-offset alignment rules to satisfy, no >4 GiB descriptor
// range (maxStorageBufferRange caps a binding; the E4B blob is 5.3 GiB).
static int import_blob(const struct gguf_context *ctx) {
    // extension present?
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(g.phys, NULL, &n, NULL);
    VkExtensionProperties *ext = malloc(n * sizeof(*ext));
    if (!ext) return -1;
    vkEnumerateDeviceExtensionProperties(g.phys, NULL, &n, ext);
    int have = 0;
    for (uint32_t i = 0; i < n; i++)
        if (strcmp(ext[i].extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) == 0)
            have = 1;
    free(ext);
    if (!have) return -1;

    VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT,
    };
    VkPhysicalDeviceProperties2 p2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &hp,
    };
    vkGetPhysicalDeviceProperties2(g.phys, &p2);
    size_t align = (size_t)hp.minImportedHostPointerAlignment;

    // gguf.c page-aligns and page-pads the blob for exactly this moment; a
    // driver wanting more than 4 KiB alignment loses the zero-copy path.
    g.blob = ctx->data;
    g.blob_size = ctx->data_size;
    if (!g.blob || align == 0 || ((uintptr_t)g.blob % align) != 0) return -1;
    g.imp_size = (g.blob_size + align - 1) / align * align;
    if (align > 4096 && g.imp_size > ((g.blob_size + 4095) & ~(size_t)4095))
        g.imp_size = g.blob_size / align * align;   // never import past the padded alloc
    if (g.imp_size == 0) return -1;

    PFN_vkGetMemoryHostPointerPropertiesEXT get_props =
        (PFN_vkGetMemoryHostPointerPropertiesEXT)
        vkGetDeviceProcAddr(g.dev, "vkGetMemoryHostPointerPropertiesEXT");
    if (!get_props) return -1;
    VkMemoryHostPointerPropertiesEXT mp = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
    };
    if (get_props(g.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                  (void *)g.blob, &mp) != VK_SUCCESS)
        return -1;

    // window geometry: size ≤ min(maxStorageBufferRange, 2 GiB), stride = half
    size_t wsize = g.max_range;
    if (wsize > (size_t)1 << 31) wsize = (size_t)1 << 31;
    g.win_stride = wsize / 2;
    if (g.win_stride == 0) return -1;

    // create the window buffers first: their memoryTypeBits constrain the alloc
    uint32_t type_bits = mp.memoryTypeBits;
    g.nwin = 0;
    for (size_t off = 0; off < g.imp_size && g.nwin < NWIN; off += g.win_stride) {
        size_t sz = g.imp_size - off < wsize ? g.imp_size - off : wsize;
        VkExternalMemoryBufferCreateInfo xi = {
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
        };
        VkBufferCreateInfo bi = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &xi,
            .size = sz, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        if (vkCreateBuffer(g.dev, &bi, NULL, &g.win[g.nwin].buf) != VK_SUCCESS) goto fail;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(g.dev, g.win[g.nwin].buf, &req);
        type_bits &= req.memoryTypeBits;
        if (req.size > g.imp_size - off) {   // driver slack past the import: shrink is
            vkDestroyBuffer(g.dev, g.win[g.nwin].buf, NULL);  // not expressible — give up
            goto fail;
        }
        g.win[g.nwin].size = sz;
        g.nwin++;
    }
    if (!type_bits) goto fail;
    int idx = find_mem_type(type_bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (idx < 0) idx = find_mem_type(type_bits, 0);
    if (idx < 0) goto fail;

    VkImportMemoryHostPointerInfoEXT imp = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
        .pHostPointer = (void *)g.blob,
    };
    VkMemoryAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &imp,
        .allocationSize = g.imp_size, .memoryTypeIndex = (uint32_t)idx,
    };
    if (vkAllocateMemory(g.dev, &ai, NULL, &g.imp) != VK_SUCCESS) goto fail;
    for (int i = 0; i < g.nwin; i++)
        if (vkBindBufferMemory(g.dev, g.win[i].buf, g.imp, (VkDeviceSize)i * g.win_stride)
            != VK_SUCCESS)
            goto fail;
    g.imported = 1;
    return 0;

fail:
    for (int i = 0; i < g.nwin; i++)
        if (g.win[i].buf) { vkDestroyBuffer(g.dev, g.win[i].buf, NULL); g.win[i].buf = VK_NULL_HANDLE; }
    g.nwin = 0;
    if (g.imp) { vkFreeMemory(g.dev, g.imp, NULL); g.imp = VK_NULL_HANDLE; }
    return -1;
}

// ---- weight residency -------------------------------------------------------

// Where tensor `t` lives on the GPU: the buffer to bind and the byte offset to
// pass the shader. Zero-copy: window lookup. Upload path: per-tensor buffer,
// created and filled on first use (cached by data pointer).
static int weight_locate(const struct gguf_tensor *t, size_t bytes,
                         VkBuffer *buf, uint32_t *base) {
    if (g.imported) {
        size_t off = (size_t)((const uint8_t *)t->data - g.blob);
        if (off < g.imp_size && bytes <= g.win_stride) {
            int wi = (int)(off / g.win_stride);
            if (wi < g.nwin && off - (size_t)wi * g.win_stride + bytes <= g.win[wi].size) {
                *buf  = g.win[wi].buf;
                *base = (uint32_t)(off - (size_t)wi * g.win_stride);
                return 0;
            }
        }
        // falls through: tensor past the import cut or over a gigabyte — upload it
    }
    uint32_t h = (uint32_t)(((uintptr_t)t->data >> 4) * 2654435761u) & (NSLOTS - 1);
    for (uint32_t i = 0; i < NSLOTS; i++, h = (h + 1) & (NSLOTS - 1)) {
        if (g.slots[h].key == t->data) { *buf = g.slots[h].buf; *base = 0; return 0; }
        if (g.slots[h].key == NULL) {
            void *map = NULL;
            if (bytes > g.max_range) return -1;
            if (make_buffer(bytes, &g.slots[h].buf, &g.slots[h].mem, &map, BUF_GPU_READ) != 0)
                return -1;
            memcpy(map, t->data, bytes);
            vkUnmapMemory(g.dev, g.slots[h].mem);
            g.slots[h].key = t->data;
            *buf = g.slots[h].buf; *base = 0;
            return 0;
        }
    }
    return -1;   // table full — cannot happen at ~400 tensors
}

// ---- pipelines ---------------------------------------------------------------

static VkPipeline make_pipe(const uint32_t *spv, size_t bytes) {
    VkShaderModuleCreateInfo si = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = bytes, .pCode = spv,
    };
    VkShaderModule mod;
    if (vkCreateShaderModule(g.dev, &si, NULL, &mod) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkComputePipelineCreateInfo pi = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "main",
        },
        .layout = g.pl,
    };
    VkPipeline p = VK_NULL_HANDLE;
    vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &pi, NULL, &p);
    vkDestroyShaderModule(g.dev, mod, NULL);
    return p;
}

// ---- init / teardown ---------------------------------------------------------

int vkc_ready(void) { return g.ready == 1; }

void vkc_destroy(void) {
    if (!g.dev) { if (g.inst) vkDestroyInstance(g.inst, NULL); memset(&g, 0, sizeof g); return; }
    vkDeviceWaitIdle(g.dev);
    for (int i = 0; i < NSLOTS; i++)
        if (g.slots[i].key) {
            vkDestroyBuffer(g.dev, g.slots[i].buf, NULL);
            vkFreeMemory(g.dev, g.slots[i].mem, NULL);
        }
    for (int i = 0; i < g.nwin; i++) vkDestroyBuffer(g.dev, g.win[i].buf, NULL);
    if (g.imp)  vkFreeMemory(g.dev, g.imp, NULL);
    if (g.xbuf) { vkDestroyBuffer(g.dev, g.xbuf, NULL); vkFreeMemory(g.dev, g.xmem, NULL); }
    if (g.obuf) { vkDestroyBuffer(g.dev, g.obuf, NULL); vkFreeMemory(g.dev, g.omem, NULL); }
    for (int i = 0; i < NTYPES; i++)
        if (g.pipe[i]) vkDestroyPipeline(g.dev, g.pipe[i], NULL);
    if (g.dpool) vkDestroyDescriptorPool(g.dev, g.dpool, NULL);
    if (g.pl)    vkDestroyPipelineLayout(g.dev, g.pl, NULL);
    if (g.dsl)   vkDestroyDescriptorSetLayout(g.dev, g.dsl, NULL);
    if (g.fence) vkDestroyFence(g.dev, g.fence, NULL);
    if (g.pool)  vkDestroyCommandPool(g.dev, g.pool, NULL);
    vkDestroyDevice(g.dev, NULL);
    vkDestroyInstance(g.inst, NULL);
    memset(&g, 0, sizeof g);
}

int vkc_init(const struct gguf_context *ctx) {
    if (g.ready) return g.ready == 1 ? 0 : -1;
    g.ready = -1;                                  // pessimistic until the end

    const char *layers[1];
    uint32_t nlayers = 0;
    if (getenv("LG_VK_VALIDATE")) layers[nlayers++] = "VK_LAYER_KHRONOS_validation";
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "little-gemma", .apiVersion = VK_API_VERSION_1_1,
    };
    VkInstanceCreateInfo ii = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
        .enabledLayerCount = nlayers, .ppEnabledLayerNames = layers,
    };
    if (vkCreateInstance(&ii, NULL, &g.inst) != VK_SUCCESS) {
        fprintf(stderr, "vulkan: no instance (loader/driver missing?) — staying on host matmul\n");
        return -1;
    }

    // physical device: LG_VK_DEVICE=n overrides; else discrete > integrated > rest
    uint32_t nphys = 0;
    vkEnumeratePhysicalDevices(g.inst, &nphys, NULL);
    if (nphys == 0) { fprintf(stderr, "vulkan: no devices\n"); return -1; }
    VkPhysicalDevice phys[16];
    if (nphys > 16) nphys = 16;
    vkEnumeratePhysicalDevices(g.inst, &nphys, phys);
    const char *want = getenv("LG_VK_DEVICE");
    int best = -1, best_score = -1;
    for (uint32_t i = 0; i < nphys; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(phys[i], &p);
        int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   ? 3
                  : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 1;
        if (want ? (uint32_t)atoi(want) == i : score > best_score) { best = (int)i; best_score = score; }
    }
    if (best < 0) return -1;
    g.phys = phys[best];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g.phys, &props);
    vkGetPhysicalDeviceMemoryProperties(g.phys, &g.mem);
    g.max_range = props.limits.maxStorageBufferRange;

    // compute queue family
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &nq, NULL);
    VkQueueFamilyProperties qf[32];
    if (nq > 32) nq = 32;
    vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &nq, qf);
    g.qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g.qfam = i; break; }
    if (g.qfam == UINT32_MAX) { fprintf(stderr, "vulkan: no compute queue\n"); return -1; }

    // device (+ the import extension when present — checked again inside import_blob)
    uint32_t next = 0;
    vkEnumerateDeviceExtensionProperties(g.phys, NULL, &next, NULL);
    VkExtensionProperties *ep = malloc(next * sizeof(*ep));
    const char *dev_ext[1];
    uint32_t ndev_ext = 0;
    if (ep) {
        vkEnumerateDeviceExtensionProperties(g.phys, NULL, &next, ep);
        for (uint32_t i = 0; i < next; i++)
            if (strcmp(ep[i].extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) == 0)
                dev_ext[ndev_ext++] = VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME;
        free(ep);
    }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = g.qfam, .queueCount = 1, .pQueuePriorities = &prio,
    };
    VkDeviceCreateInfo di = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi,
        .enabledExtensionCount = ndev_ext, .ppEnabledExtensionNames = dev_ext,
    };
    if (vkCreateDevice(g.phys, &di, NULL, &g.dev) != VK_SUCCESS) {
        fprintf(stderr, "vulkan: device creation failed\n");
        return -1;
    }
    vkGetDeviceQueue(g.dev, g.qfam, 0, &g.queue);

    VkCommandPoolCreateInfo pi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = g.qfam,
    };
    VkCommandBufferAllocateInfo cbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
    };
    VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateCommandPool(g.dev, &pi, NULL, &g.pool) != VK_SUCCESS) return -1;
    cbi.commandPool = g.pool;
    if (vkAllocateCommandBuffers(g.dev, &cbi, &g.cb) != VK_SUCCESS) return -1;
    if (vkCreateFence(g.dev, &fi, NULL, &g.fence) != VK_SUCCESS) return -1;

    // one layout for every pipeline: W, x, out storage buffers + 20B push consts
    VkDescriptorSetLayoutBinding binds[3];
    for (int i = 0; i < 3; i++)
        binds[i] = (VkDescriptorSetLayoutBinding){
            .binding = (uint32_t)i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    VkDescriptorSetLayoutCreateInfo dli = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = binds,
    };
    if (vkCreateDescriptorSetLayout(g.dev, &dli, NULL, &g.dsl) != VK_SUCCESS) return -1;
    VkPushConstantRange pc = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 5 * sizeof(uint32_t),
    };
    VkPipelineLayoutCreateInfo pli = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &g.dsl,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &pc,
    };
    if (vkCreatePipelineLayout(g.dev, &pli, NULL, &g.pl) != VK_SUCCESS) return -1;

    VkDescriptorPoolSize dps = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3 };
    VkDescriptorPoolCreateInfo dpi = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps,
    };
    if (vkCreateDescriptorPool(g.dev, &dpi, NULL, &g.dpool) != VK_SUCCESS) return -1;
    VkDescriptorSetAllocateInfo dsi = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = g.dpool, .descriptorSetCount = 1, .pSetLayouts = &g.dsl,
    };
    if (vkAllocateDescriptorSets(g.dev, &dsi, &g.ds) != VK_SUCCESS) return -1;

    g.pipe[GGML_TYPE_F32]  = make_pipe(spv_f32,  sizeof spv_f32);
    g.pipe[GGML_TYPE_F16]  = make_pipe(spv_f16,  sizeof spv_f16);
    g.pipe[GGML_TYPE_Q4_0] = make_pipe(spv_q4_0, sizeof spv_q4_0);
    g.pipe[GGML_TYPE_Q8_0] = make_pipe(spv_q8_0, sizeof spv_q8_0);
    g.pipe[GGML_TYPE_Q4_K] = make_pipe(spv_q4_K, sizeof spv_q4_K);
    g.pipe[GGML_TYPE_Q6_K] = make_pipe(spv_q6_K, sizeof spv_q6_K);
    g.pipe[GGML_TYPE_BF16] = make_pipe(spv_bf16, sizeof spv_bf16);

    // LG_VK_NO_IMPORT forces the upload path — the A/B for the question the
    // memory-type comments above answer (snooped host pages vs GTT).
    int zero_copy = !getenv("LG_VK_NO_IMPORT") && import_blob(ctx) == 0;
    fprintf(stderr, "vulkan: %s — weights %s\n", props.deviceName,
            zero_copy ? "zero-copy (GGUF blob imported in place)"
                      : "uploaded per tensor (second copy of the model in RAM)");
    g.ready = 1;
    return 0;
}

// ---- the matmul --------------------------------------------------------------

static int io_reserve(size_t xbytes, size_t obytes) {
    if (xbytes > g.xcap) {
        size_t cap = g.xcap ? g.xcap : 1 << 18;
        while (cap < xbytes) cap *= 2;
        if (g.xbuf) {
            vkDeviceWaitIdle(g.dev);
            vkDestroyBuffer(g.dev, g.xbuf, NULL);
            vkFreeMemory(g.dev, g.xmem, NULL);
            g.xbuf = VK_NULL_HANDLE; g.xmem = VK_NULL_HANDLE;
        }
        if (make_buffer(cap, &g.xbuf, &g.xmem, &g.xmap, BUF_GPU_READ) != 0) { g.xcap = 0; return -1; }
        g.xcap = cap;
    }
    if (obytes > g.ocap) {
        size_t cap = g.ocap ? g.ocap : 1 << 20;
        while (cap < obytes) cap *= 2;
        if (g.obuf) {
            vkDeviceWaitIdle(g.dev);
            vkDestroyBuffer(g.dev, g.obuf, NULL);
            vkFreeMemory(g.dev, g.omem, NULL);
            g.obuf = VK_NULL_HANDLE; g.omem = VK_NULL_HANDLE;
        }
        if (make_buffer(cap, &g.obuf, &g.omem, &g.omap, BUF_CPU_READ) != 0) { g.ocap = 0; return -1; }
        g.ocap = cap;
    }
    return 0;
}

int vkc_matmul(float *out, const struct gguf_tensor *t, const float *x, int k, int m) {
    if (g.ready != 1 || (uint32_t)t->type >= NTYPES || !g.pipe[t->type]) return -1;
    if (k <= 0 || m <= 0 || k % 32 != 0) return -1;
    int blck = ggml_blck_size(t->type);
    if (blck == 0 || k % blck != 0) return -1;

    size_t bytes = (size_t)(k / blck) * ggml_type_size(t->type) * (size_t)m;
    VkBuffer wbuf;
    uint32_t wbase;
    if (weight_locate(t, bytes, &wbuf, &wbase) != 0) return -1;
    if (io_reserve((size_t)k * 4, (size_t)m * 4) != 0) return -1;

    memcpy(g.xmap, x, (size_t)k * 4);       // host-coherent: visible at submit

    VkDescriptorBufferInfo bw = { .buffer = wbuf,   .offset = 0, .range = VK_WHOLE_SIZE };
    VkDescriptorBufferInfo bx = { .buffer = g.xbuf, .offset = 0, .range = VK_WHOLE_SIZE };
    VkDescriptorBufferInfo bo = { .buffer = g.obuf, .offset = 0, .range = VK_WHOLE_SIZE };
    VkWriteDescriptorSet wr[3];
    VkDescriptorBufferInfo *bi[3] = { &bw, &bx, &bo };
    for (int i = 0; i < 3; i++)
        wr[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g.ds,
            .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = bi[i],
        };
    vkUpdateDescriptorSets(g.dev, 3, wr, 0, NULL);   // set idle: fence was waited

    uint32_t pcv[5] = { (uint32_t)k, (uint32_t)m, wbase, 0, 0 };
    uint32_t gx = m < 32768 ? (uint32_t)m : 32768u;
    uint32_t gy = ((uint32_t)m + 32767u) / 32768u;

    VkCommandBufferBeginInfo bgi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(g.cb, &bgi);
    vkCmdBindPipeline(g.cb, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe[t->type]);
    vkCmdBindDescriptorSets(g.cb, VK_PIPELINE_BIND_POINT_COMPUTE, g.pl, 0, 1, &g.ds, 0, NULL);
    vkCmdPushConstants(g.cb, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pcv, pcv);
    vkCmdDispatch(g.cb, gx, gy, 1);
    VkMemoryBarrier mb = {                    // shader writes -> host reads
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(g.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkEndCommandBuffer(g.cb);

    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &g.cb,
    };
    vkResetFences(g.dev, 1, &g.fence);
    if (vkQueueSubmit(g.queue, 1, &si, g.fence) != VK_SUCCESS) return -1;
    if (vkWaitForFences(g.dev, 1, &g.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) return -1;

    memcpy(out, g.omap, (size_t)m * 4);
    return 0;
}
