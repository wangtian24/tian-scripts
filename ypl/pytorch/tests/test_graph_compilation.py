import random
import time

from tqdm import trange

from ypl.pytorch.model.categorizer import PromptTopicDifficultyModel


def test_categorizer_cuda_graph() -> None:
    random.seed(0)
    model = PromptTopicDifficultyModel("roberta-base", label_map=dict(a=0, b=1, c=2)).cuda()

    # Normal
    a = time.time()

    for _ in trange(1000):
        model.categorize("a" * random.randint(1, 512))

    normal_time = time.time() - a

    # TorchDynamo
    model = model.compile()

    a = time.time()

    for _ in trange(1000):
        model.categorize("a" * random.randint(1, 512))

    dynamo_time = time.time() - a

    # CUDA graphs
    model = PromptTopicDifficultyModel("roberta-base", label_map=dict(a=0, b=1, c=2)).cuda()
    model.compile_cuda_graphs()

    a = time.time()

    for _ in trange(1000):
        model.categorize("a" * random.randint(1, 512))

    cuda_graph_time = time.time() - a

    print(f"Normal: {normal_time}")
    print(f"TorchDynamo: {dynamo_time}")
    print(f"CUDA graphs: {cuda_graph_time}")

    assert cuda_graph_time < normal_time / 2
    assert cuda_graph_time < dynamo_time / 2
    assert dynamo_time < normal_time
