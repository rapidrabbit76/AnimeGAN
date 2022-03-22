import pytest


@pytest.mark.parametrize("train", [True, False])
def test_gen_forward(batch, gen, train):
    gen = gen.train(train)
    x, y = batch
    y_ = gen(x)
    assert y_.shape == y.shape


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
