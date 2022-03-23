import pytest
import losses


def test_content_loss(batch, encoder):
    _y, y = batch
    loss = losses.content_loss(encoder, y, _y)
    assert list(loss.shape) == []


def test_gram_matrix(batch):
    _y, y = batch
    y_g = losses.gram_matrix(y)
    _y_g = losses.gram_matrix(_y)
    assert _y_g.shape == y_g.shape


def test_style_loss(batch):
    _y, y = batch
    loss = losses.style_loss(_y, y)
    assert list(loss.shape) == []


def test_color_loss(batch):
    _y, y = batch
    loss = losses.color_loss(_y, y)
    assert list(loss.shape) == []
