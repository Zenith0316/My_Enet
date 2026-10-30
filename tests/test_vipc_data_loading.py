"""Use a tiny on-disk ViPC sample to exercise paths, encodings and view selection."""

import pickle

import numpy as np
import pytest
import torch
from PIL import Image

from utils.ViPCdataloader import ViPCDataLoader


@pytest.fixture
def sample(tmp_path):
    root = tmp_path / '数据.with.dots'
    points = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float32)
    for subtree in ('ShapeNetViPC-Partial', 'ShapeNetViPC-GT', 'ShapeNetViPC-View'):
        directory = root / subtree / '02691156' / 'object'
        directory.mkdir(parents=True)
        if subtree.endswith('View'):
            directory = directory / 'rendering'
            directory.mkdir()
            np.savetxt(directory / 'rendering_metadata.txt', np.zeros((24, 4)), encoding='utf-8-sig')
            for view in ('00', '17'):
                Image.new('RGBA', (32, 32), (10 + int(view), 20, 30, 255)).save(directory / (view + '.png'))
        else:
            for view in ('00', '17'):
                with (directory / (view + '.dat')).open('wb') as file:
                    pickle.dump(points, file, protocol=2)
    return root, points


@pytest.mark.parametrize('ending', ['', '\n', '\r\n'])
@pytest.mark.parametrize('aligned', [False, True])
def test_read_sample_without_newline_dependency(sample, tmp_path, monkeypatch, ending, aligned):
    root, points = sample
    index = tmp_path / '索引.txt'
    index.write_bytes(('\ufeff\r\n02691156\\object\\17' + ending).encode('utf-8'))
    monkeypatch.setattr('utils.ViPCdataloader.random.randint', lambda low, high: 0)
    dataset = ViPCDataLoader(index, root, 'test', pc_input_num=8,
                            view_align=aligned, category='plane')
    image, gt, partial = dataset[0]
    assert len(dataset) == 1
    assert image.shape == (3, 224, 224)
    assert image[0, 0, 0].item() == pytest.approx((27 if aligned else 10) / 255)
    assert gt.dtype == partial.dtype == torch.float32
    np.testing.assert_allclose(gt.numpy(), points)
    np.testing.assert_allclose(partial.numpy(), np.repeat(points, 3, axis=0)[:8], atol=1e-6)


@pytest.mark.parametrize('entry', ['02691156/object/24', '02691156/object', '../object/00'])
def test_invalid_index_has_line_number(sample, tmp_path, entry):
    root, _ = sample
    index = tmp_path / 'invalid.txt'
    index.write_text(entry, encoding='utf-8')
    with pytest.raises(ValueError, match='invalid.txt:1:'):
        ViPCDataLoader(index, root, 'test')


def test_empty_partial_has_file_context(sample, tmp_path):
    root, _ = sample
    index = tmp_path / 'index.txt'
    index.write_text('02691156/object/17', encoding='utf-8')
    with (root / 'ShapeNetViPC-Partial/02691156/object/17.dat').open('wb') as file:
        pickle.dump(np.empty((0, 3)), file)
    dataset = ViPCDataLoader(index, root, 'test', view_align=True)
    with pytest.raises(ValueError, match='17.dat'):
        dataset[0]
