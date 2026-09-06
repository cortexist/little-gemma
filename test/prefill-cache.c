// Differential CUDA prefill gate (requires a GGUF and a 2k-6k-token text file).
// Build explicitly: cmake --build build --target prefill_cache_test
// Run in separate processes, then compare stdout:
//   LG_PREFILL_FULL=1 build/prefill_cache_test MODEL TEXT > full.txt
//   build/prefill_cache_test MODEL TEXT > cache.txt
//   diff -u full.txt cache.txt
// Hashes EVERY allocated K/V byte, including padding and wrapped ring slots,
// after clearing unused storage. Also compares 12 generated token IDs per case.
// Synthetic media rows exercise bidirectional masking without an encoder/model.
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "gguf.h"
#include "model.h"
#include "tokenizer.h"

int (*model_pick)(const float *, int) = NULL;
#ifdef LG_MTP_N_MAX // runtime-MTP builds normally define this in run.c
int g_mtp_n = LG_MTP_N;
#endif

#define CU(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);} } while(0)
static void clear_cache(struct kvcache *kv) {
    for(int l=0;l<kv->n_layer;l++) if(kv->k[l]) {
        size_t n=(size_t)kv->seq[l]*kv->kv_dim[l]*(kv->f16[l]?2:4);
        CU(cudaMemset(kv->k[l],0,n)); CU(cudaMemset(kv->v[l],0,n));
    }
    CU(cudaDeviceSynchronize());
}
static void report(struct model *m,struct kvcache *kv,const char *name,int n,int next) {
    CU(cudaDeviceSynchronize());
    uint64_t hash=UINT64_C(14695981039346656037);
    for(int l=0;l<kv->n_layer;l++) if(kv->k[l]) {
        size_t size=(size_t)kv->seq[l]*kv->kv_dim[l]*(kv->f16[l]?2:4);
        unsigned char *host=malloc(size); if(!host)exit(2);
        for(int v=0;v<2;v++) {
            CU(cudaMemcpy(host,v?kv->v[l]:kv->k[l],size,cudaMemcpyDeviceToHost));
            for(size_t i=0;i<size;i++){hash^=host[i];hash*=UINT64_C(1099511628211);}
        }
        free(host);
    }
    printf("%s %d cache=%016llx tokens=",name,n,(unsigned long long)hash);
    for(int i=0;i<12;i++){next=model_forward_next(m,kv,next,n+i);printf("%d,",next);}
    printf("\n");fflush(stdout);
}
int main(int argc,char **argv) {
    if(argc!=3){fprintf(stderr,"usage: %s MODEL.gguf LONG-PROMPT.txt\n",argv[0]);return 2;}
    struct gguf_context *ctx=load_gguf(argv[1]); if(!ctx)return 2;
    struct model m; if(model_init(&m,ctx)!=0)return 2;
    struct tokenizer *tk=tokenizer_init(ctx);if(!tk)return 2;
    struct kvcache kv;if(kvcache_init(&kv,&m,8192)!=0)return 2;
    FILE *fp=fopen(argv[2],"rb");if(!fp)return 2;
    if(fseek(fp,0,SEEK_END)!=0)return 2;
    long len=ftell(fp);if(len<0)return 2;rewind(fp);
    char *text=calloc((size_t)len+1,1);if(!text)return 2;
    if(fread(text,1,len,fp)!=(size_t)len)return 2;
    fclose(fp);
    int tokens[7000];int n=tokenizer_encode(tk,text,tokens,7000);free(text);
    if(n<2000||n>6000)return 2;
    // Cold chunk exercises the allocation/warmup exception as well as truncation.
    clear_cache(&kv);model_prefill(&m,&kv,tokens,930,0);
    report(&m,&kv,"cold",930,tokens[930]);
    clear_cache(&kv);model_prefill(&m,&kv,tokens,n-1,0);
    report(&m,&kv,"text",n-1,tokens[n-1]);
    clear_cache(&kv);model_prefill_mixed(&m,&kv,NULL,tokens,n-1,0);
    report(&m,&kv,"mixed-text",n-1,tokens[n-1]);
    float *rows=malloc((size_t)66*m.cfg.n_embd*sizeof(float));if(!rows)return 2;
    for(int i=0;i<66*m.cfg.n_embd;i++)rows[i]=(float)((i*17)%101-50)/64.f;
    int ids[7000];memcpy(ids,tokens,n*sizeof(int));
    for(int i=0;i<66;i++)ids[n/2+i]=-i-1;
    clear_cache(&kv);model_prefill_mixed(&m,&kv,rows,ids,n-1,0);
    report(&m,&kv,"mixed-media",n-1,tokens[n-1]);
    clear_cache(&kv);model_prefill_embd(&m,&kv,rows,66,0);
    report(&m,&kv,"embeddings",66,tokens[0]);
    free(rows);tokenizer_free(tk);kvcache_free(&kv);model_free(&m);free_gguf(ctx);
    return 0;
}
