import os
import pytest
import torch
import onnxruntime


@pytest.mark.parametrize("train", [True, False])
def test_gen_forward(batch, gen, train):
    gen = gen.train(train)
    x, y = batch
    y_ = gen(x)
    assert y_.shape == y.shape


@pytest.fixture(scope="module")
def example_inputs(batch):
    return batch[0]


def test_to_torchscript(save_dir, gen, example_inputs):
    save_dir = save_dir.name
    torchscript_path = os.path.join(save_dir, "AnimeGAN.pt.zip")
    gen.to_torchscript(torchscript_path, "trace", example_inputs)
    assert os.path.exists(torchscript_path)


def test_load_torchscript(save_dir, example_inputs):
    save_dir = save_dir.name
    torchscript_path = os.path.join(save_dir, "AnimeGAN.pt.zip")
    model = torch.jit.load(torchscript_path)
    x = model(example_inputs)
    assert x.shape == example_inputs.shape


def test_to_onnx(save_dir, gen, example_inputs):
    save_dir = save_dir.name
    onnx_path = os.path.join(save_dir, "AnimeGAN.onnx")
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {
        input_names[0]: {0: "batch_size", 1: "c", 2: "h", 3: "w"},
        output_names[0]: {0: "batch_size", 1: "c", 2: "h", 3: "w"},
    }
    gen.to_onnx(
        file_path=onnx_path,
        input_sample=example_inputs,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=12,
        dynamic_axes=dynamic_axes,
    )
    assert os.path.exists(onnx_path)


def test_load_onnx(save_dir, example_inputs):
    save_dir = save_dir.name
    onnx_path = os.path.join(save_dir, "AnimeGAN.onnx")
    session = onnxruntime.InferenceSession(onnx_path)
    example_inputs = torch.rand([1, 3, 512, 512]).numpy()

    inputs_tag = session.get_inputs()
    outputs_tag = session.get_outputs()
    inputs = {inputs_tag[0].name: example_inputs}
    output = session.run([outputs_tag[0].name], inputs)[0]
    assert output.shape == example_inputs.shape


def test_disc_forward(batch, disc):
    x, y = batch
    assert x.shape == y.shape
    b, _, w, h = x.shape

    logits = disc(x)
    assert list(logits.shape) == [b, 1, w // 8, h // 8]
    logits = disc(y)
    assert list(logits.shape) == [b, 1, w // 8, h // 8]


def test_encoder_forward(batch, encoder):
    x, y = batch
    assert x.shape == y.shape
    b, _, w, h = x.shape
    x_f = encoder(x)
    assert list(x_f.shape) == [b, 512, 32, 32]
    y_f = encoder(y)
    assert list(y_f.shape) == [b, 512, 32, 32]
