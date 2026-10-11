#include "decision-head.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)
#define CLOSE(a,b) CHECK(fabsf((a)-(b)) < 2e-6f)

int main(void) {
    const char *labels[] = {"route", "think", "hold"};
    float mean[] = {1, 2}, scale[] = {2, 4};
    float w0[] = {1,0, 0,1, 1,1}, b0[] = {0,0,1};
    float w1[] = {2,0, 0,2, -1,-1}, b1[] = {0,0,-1};
    struct decision_head_member members[] = {{NULL,NULL,w0,b0}, {NULL,NULL,w1,b1}};
    struct decision_head_spec spec = {
        .architecture = DECISION_HEAD_LINEAR,
        .features = {2, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                     "test_features_v1", "test_boundary", -1, 0},
        .output_dim = 3, .ensemble_size = 2, .clip = 2, .temperature = 2,
        .mean = mean, .scale = scale, .members = members, .labels = labels,
        .task = "route_selection", .source_contract_json = "{}"
    };
    char error[256];
    struct decision_head *head = decision_head_create(&spec, error, sizeof error);
    CHECK(head);
    CHECK(!decision_head_check_features(head, &spec.features, error, sizeof error));
    struct decision_head_features wrong = spec.features;
    wrong.layer = 0; CHECK(decision_head_check_features(head, &wrong, error, sizeof error));
    wrong = spec.features; wrong.answer_channel_seed = 1;
    CHECK(decision_head_check_features(head, &wrong, error, sizeof error));
    wrong = spec.features; wrong.capture_boundary = "later";
    CHECK(decision_head_check_features(head, &wrong, error, sizeof error));
    wrong = spec.features; wrong.model_sha256 = "different";
    CHECK(decision_head_check_features(head, &wrong, error, sizeof error));
    wrong = spec.features; wrong.input_format = "different";
    CHECK(decision_head_check_features(head, &wrong, error, sizeof error));
    wrong = spec.features; wrong.input_dim = 3;
    CHECK(decision_head_check_features(head, &wrong, error, sizeof error));
    struct decision_head_workspace *workspace = decision_head_workspace_create(head);
    struct decision_head_workspace *workspace2 = decision_head_workspace_create(head);
    CHECK(workspace && workspace2);
    w0[0] = 999; /* Owned copy must remain unchanged. */
    float x[] = {5,-10}, logits[3], probs[3]; size_t choice;
    CHECK(!decision_head_predict(head, workspace, x, 2, logits, probs, 3, &choice, error, sizeof error));
    CHECK(choice == 0);
    CLOSE(logits[0],3); CLOSE(logits[1],-3); CLOSE(logits[2],0);
    CLOSE(probs[0],0.78559703f); CLOSE(probs[1],0.03911257f); CLOSE(probs[2],0.17529039f);
    CHECK(!strcmp(decision_head_get_spec(head)->labels[choice], "route"));
    CHECK(!decision_head_predict(head, workspace2, mean, 2, logits, probs, 3, &choice, error, sizeof error));
    CHECK(choice == 0); CLOSE(probs[0],1.0f/3); CLOSE(probs[1],1.0f/3);
    CHECK(!decision_head_predict(head, workspace, x, 2, logits, probs, 3, &choice, error, sizeof error));
    CLOSE(logits[0],3);
    logits[0] = 123; probs[0] = 456;
    CHECK(decision_head_predict(head, workspace, x, 1, logits, probs, 3, &choice, error, sizeof error));
    CHECK(choice == SIZE_MAX && logits[0] == 123 && probs[0] == 456);
    CHECK(decision_head_predict(head, workspace, x, 2, logits, probs, 2, &choice, error, sizeof error));
    x[0] = NAN;
    CHECK(decision_head_predict(head, workspace, x, 2, logits, probs, 3, &choice, error, sizeof error));
    CHECK(choice == SIZE_MAX && logits[0] == 123 && probs[0] == 456);
    decision_head_workspace_free(workspace2);

    /* Independent analytic tanh example: tanh(atanh(.5)) = .5. */
    const char *binary_labels[] = {"accept", "reject"};
    float m[] = {0}, sc[] = {1}, hw[] = {1,-1}, hb[] = {0,0};
    float ow[] = {1,-1, -1,1}, ob[] = {0,0};
    struct decision_head_member mlp_member = {hw,hb,ow,ob};
    struct decision_head_spec mlp = spec;
    mlp.architecture = DECISION_HEAD_MLP; mlp.features.input_dim = 1; mlp.hidden_dim = 2;
    mlp.output_dim = 2; mlp.ensemble_size = 1; mlp.temperature = 1;
    mlp.mean = m; mlp.scale = sc; mlp.members = &mlp_member; mlp.labels = binary_labels;
    struct decision_head *other = decision_head_create(&mlp, error, sizeof error);
    CHECK(other);
    struct decision_head_workspace *owork = decision_head_workspace_create(other); CHECK(owork);
    float mx = 0.5493061443f;
    CHECK(!decision_head_predict(other, owork, &mx, 1, logits, probs, 2, &choice, error, sizeof error));
    CLOSE(logits[0],1); CLOSE(logits[1],-1); CLOSE(probs[0],0.88079708f); CHECK(choice == 0);
    CHECK(decision_head_predict(other, workspace, &mx, 1, logits, probs, 2, &choice, error, sizeof error));
    decision_head_workspace_free(workspace); decision_head_free(head);
    decision_head_workspace_free(owork); decision_head_free(other);

    /* Almost-equal logits can round to identical FP32 probabilities. The
     * reference selects the first probability, not the larger raw logit. */
    ob[1] = 1e-8f; mx = 0;
    other = decision_head_create(&mlp, error, sizeof error); CHECK(other);
    owork = decision_head_workspace_create(other); CHECK(owork);
    CHECK(!decision_head_predict(other, owork, &mx, 1, logits, probs, 2, &choice, error, sizeof error));
    CHECK(logits[1] > logits[0]); CHECK(probs[0] == probs[1]); CHECK(choice == 0);
    decision_head_workspace_free(owork); decision_head_free(other);

    scale[0] = 0; CHECK(!decision_head_create(&spec, error, sizeof error)); scale[0] = 2;
    w0[0] = NAN; CHECK(!decision_head_create(&spec, error, sizeof error)); w0[0] = 1;
    spec.temperature = 0; CHECK(!decision_head_create(&spec, error, sizeof error)); spec.temperature = 2;
    labels[1] = "route"; CHECK(!decision_head_create(&spec, error, sizeof error)); labels[1] = "think";
    spec.features.input_dim = SIZE_MAX; CHECK(!decision_head_create(&spec, error, sizeof error));
    puts("decision head: analytic inference, generic labels, ownership and contracts OK");
    return 0;
}
