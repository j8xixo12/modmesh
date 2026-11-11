# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Static GLB input must preserve instances and reject malformed geometry."""

import copy
import json
import struct
import zlib

import numpy as np
import pytest

from solvcon.apps.modelview import _gltf, _scene


def fixture():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype='<f4')
    normals = np.tile(np.array([0, -1, 1], dtype='<f4'), (3, 1))
    normals /= np.sqrt(2)
    vertices = np.concatenate([positions, normals], axis=1)
    binary = b'\0' * 8 + vertices.tobytes() + bytes([0, 1, 2])
    doc = dict(asset=dict(version='2.0'), scene=0,
               scenes=[dict(nodes=[0])],
               nodes=[dict(name='Assembly', translation=[1, 2, 3],
                           scale=[2, 3, 4], children=[1, 2]),
                      dict(name='Part', mesh=0),
                      dict(name='Part', mesh=0, translation=[2, 0, 0])],
               meshes=[dict(primitives=[dict(
                   attributes=dict(POSITION=0, NORMAL=1), indices=2)])],
               buffers=[dict(byteLength=len(binary))],
               bufferViews=[dict(buffer=0, byteOffset=8, byteLength=72,
                                 byteStride=24),
                            dict(buffer=0, byteOffset=80, byteLength=3)],
               accessors=[dict(bufferView=0, componentType=5126,
                               type='VEC3', count=3),
                          dict(bufferView=0, byteOffset=12, componentType=5126,
                               type='VEC3', count=3),
                          dict(bufferView=1, componentType=5121,
                               type='SCALAR', count=3)])
    return doc, binary


def encode(doc, binary):
    js = json.dumps(doc).encode()
    js += b' ' * (-len(js) % 4)
    binary += b'\0' * (-len(binary) % 4)
    return (struct.pack('<III', 0x46546C67, 2, 28 + len(js) + len(binary))
            + struct.pack('<II', len(js), 0x4E4F534A) + js
            + struct.pack('<II', len(binary), 0x004E4942) + binary)


def load(tmp_path, mutate=None, data=None):
    doc, binary = fixture()
    if mutate is not None:
        mutate(doc)
    path = tmp_path / 'fixture.glb'
    path.write_bytes(encode(doc, binary) if data is None else data)
    return _gltf.load_glb(path)


def test_material_factors_survive_scene_instancing(tmp_path):
    def material(doc):
        doc['materials'] = [dict(pbrMetallicRoughness=dict(
            roughnessFactor=0.25, metallicFactor=0.8))]
        doc['meshes'][0]['primitives'][0]['material'] = 0
    items = _scene.surfaces(load(tmp_path, material))
    assert len(items) == 2
    for item in items:
        assert item.roughness == 0.25
        assert item.metallic == 0.8


@pytest.mark.parametrize('value', [-0.1, 1.1, float('nan'), float('inf')])
def test_invalid_material_factors_rejected_before_scene_load(tmp_path, value):
    def material(doc):
        doc['materials'] = [dict(pbrMetallicRoughness=dict(
            roughnessFactor=value))]
    with pytest.raises(ValueError, match='roughnessFactor'):
        load(tmp_path, material)


def test_shared_mesh_instances_and_duplicate_names(tmp_path):
    model = load(tmp_path)
    items = _scene.surfaces(model)
    assert len(items) == 2
    assert items[0].key != items[1].key
    assert model.nodes[1].name == model.nodes[2].name
    np.testing.assert_allclose(items[0].positions,
                               [[3, 1, 2], [3, 3, 2], [7, 1, 5]])
    np.testing.assert_allclose(items[1].positions - items[0].positions,
                               [[0, 4, 0]] * 3)
    np.testing.assert_allclose(_scene.bounds(items),
                               [[3, 1, 2], [7, 7, 5]])


def test_normals_use_inverse_transpose_not_position_transform(tmp_path):
    item = _scene.surfaces(load(tmp_path))[0]
    edge1, edge2 = item.positions[1:] - item.positions[0]
    expected = np.cross(edge1, edge2)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(item.normals, [expected] * 3, atol=1e-6)


def test_mirrored_instance_reverses_winding(tmp_path):
    def mirror(doc):
        doc['nodes'][1]['scale'] = [-1, 1, 1]
    item = _scene.surfaces(load(tmp_path, mirror))[0]
    assert item.indices.tolist() == [[0, 2, 1]]
    p = item.positions[item.indices[0]]
    assert np.dot(np.cross(p[1] - p[0], p[2] - p[0]), item.normals[0]) > 0


def test_missing_normals_have_geometric_fallback(tmp_path):
    def remove(doc):
        del doc['meshes'][0]['primitives'][0]['attributes']['NORMAL']
    model = load(tmp_path, remove)
    assert 'Missing normals' in model.notices[0]
    np.testing.assert_allclose(model.meshes[0][0].normals,
                               [[0, -1 / np.sqrt(2), 1 / np.sqrt(2)]] * 3)


def test_unindexed_and_multiple_primitives(tmp_path):
    def modify(doc):
        specs = doc['meshes'][0]['primitives']
        del specs[0]['indices']
        specs.append(copy.deepcopy(specs[0]))
    model = load(tmp_path, modify)
    assert len(_scene.surfaces(model)) == 4
    assert model.meshes[0][0].indices.tolist() == [[0, 1, 2]]


@pytest.mark.parametrize('dtype,component', [('<u2', 5123), ('<u4', 5125)])
def test_larger_index_component_types(tmp_path, dtype, component):
    doc, binary = fixture()
    binary = binary[:80] + np.array([0, 1, 2], dtype=dtype).tobytes()
    doc['buffers'][0]['byteLength'] = len(binary)
    doc['bufferViews'][1]['byteLength'] = len(binary) - 80
    doc['accessors'][2]['componentType'] = component
    model = load(tmp_path, data=encode(doc, binary))
    assert model.meshes[0][0].indices.tolist() == [[0, 1, 2]]


def test_matrix_and_quaternion_have_same_rotation(tmp_path):
    doc, binary = fixture()
    doc['nodes'][1]['rotation'] = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
    model = load(tmp_path, data=encode(doc, binary))
    rotated = _scene.surfaces(model)[0].positions
    del doc['nodes'][1]['rotation']
    doc['nodes'][1]['matrix'] = [0, 1, 0, 0, -1, 0, 0, 0,
                                 0, 0, 1, 0, 0, 0, 0, 1]
    model = load(tmp_path, data=encode(doc, binary))
    np.testing.assert_allclose(rotated, _scene.surfaces(model)[0].positions)


@pytest.mark.parametrize('change', [
    lambda d: d['accessors'][0].update(count=10**12),
    lambda d: d['accessors'][0].update(byteOffset=100),
    lambda d: d['bufferViews'][0].update(byteLength=1000),
    lambda d: d['bufferViews'][0].update(byteStride=8),
    lambda d: d['accessors'][0].update(sparse={}),
    lambda d: d['accessors'][0].update(normalized=True),
    lambda d: d['buffers'][0].update(uri='external.bin'),
    lambda d: d['meshes'][0]['primitives'][0].update(mode=5),
    lambda d: d.update(extensionsRequired=['KHR_draco_mesh_compression']),
    lambda d: d['nodes'][1].update(skin=0),
    lambda d: d['nodes'][1].update(mesh=99),
    lambda d: d['nodes'][0].update(children=[1, 1]),
    lambda d: d['nodes'][1].update(children=[0]),
    lambda d: d['nodes'][1].update(matrix=[1] * 16, scale=[1, 1, 1]),
    lambda d: d['nodes'][1].update(rotation=[0, 0, 0, 0]),
    lambda d: d['scenes'][0].update(nodes=[0, 0]),
])
def test_invalid_or_unsupported_geometry_is_rejected(tmp_path, change):
    with pytest.raises(ValueError):
        load(tmp_path, change)


def test_invalid_index_rejected_before_drawing(tmp_path):
    doc, binary = fixture()
    with pytest.raises(ValueError, match='indices'):
        load(tmp_path, data=encode(doc, binary[:82] + b'\xff'))


def test_nonfinite_attribute_rejected(tmp_path):
    doc, binary = fixture()
    binary = binary[:8] + struct.pack('<f', float('nan')) + binary[12:]
    with pytest.raises(ValueError, match='Nonfinite'):
        load(tmp_path, data=encode(doc, binary))


def test_corrupt_file_lengths_rejected(tmp_path):
    doc, binary = fixture()
    good = encode(doc, binary)
    for bad in (b'glTF', good[:-1], good + b'\0', good[:12]):
        with pytest.raises(ValueError):
            load(tmp_path, data=bad)


def test_singular_transform_rejected_by_adapter(tmp_path):
    model = load(tmp_path, lambda d: d['nodes'][1].update(scale=[0, 1, 1]))
    with pytest.raises(ValueError, match='Singular'):
        _scene.surfaces(model)


def test_animations_do_not_silently_claim_playback(tmp_path):
    model = load(tmp_path, lambda d: d.update(animations=[{}]))
    assert 'Static pose only' in model.notices[0]


def png_bytes():
    def chunk(kind, data):
        return (struct.pack('>I', len(data)) + kind + data
                + struct.pack('>I', zlib.crc32(kind + data)))
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 6, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(b'\0\x20\x80\xf0\x80'))
            + chunk(b'IEND', b''))


def textured_fixture():
    doc, binary = fixture()
    binary += b'\0' * (-len(binary) % 4)
    start = len(binary)
    uv = np.array([[0, 0], [1, 0], [0, 1]], dtype='<f4')
    binary += uv.tobytes()
    doc['bufferViews'].append(dict(buffer=0, byteOffset=start, byteLength=24))
    doc['accessors'].append(dict(bufferView=2, componentType=5126,
                                 type='VEC2', count=3))
    primitive = doc['meshes'][0]['primitives'][0]
    primitive['attributes']['TEXCOORD_0'] = 3
    primitive['material'] = 0
    doc['images'] = [dict(bufferView=3, mimeType='image/png')]
    png = png_bytes()
    doc['bufferViews'].append(dict(buffer=0, byteOffset=len(binary),
                                   byteLength=len(png)))
    binary += png
    doc['buffers'][0]['byteLength'] = len(binary)
    doc['samplers'] = [dict(wrapS=33071, wrapT=33648)]
    doc['textures'] = [dict(source=0, sampler=0)]
    doc['materials'] = [dict(name='Paint', alphaMode='BLEND', alphaCutoff=0.3,
                             pbrMetallicRoughness=dict(
                                 baseColorFactor=[0.2, 0.4, 0.8, 0.6],
                                 baseColorTexture=dict(index=0)))]
    return doc, binary


def test_texture_uv_factor_and_alpha_survive_scene_conversion(tmp_path):
    doc, binary = textured_fixture()
    model = load(tmp_path, data=encode(doc, binary))
    item = _scene.surfaces(model)[0]
    assert item.texture == png_bytes()
    assert item.color == (0.2, 0.4, 0.8, 0.6)
    assert item.alpha_mode == 'BLEND'
    assert item.alpha_cutoff == 0.3
    assert (item.wrap_s, item.wrap_t) == (33071, 33648)
    np.testing.assert_array_equal(item.texcoords, [[0, 0], [1, 0], [0, 1]])


@pytest.mark.parametrize('modify', [
    lambda d: d['bufferViews'][3].update(byteLength=10**12),
    lambda d: d['images'][0].update(uri='texture.png'),
    lambda d: d['images'][0].update(mimeType='image/webp'),
    lambda d: d['textures'][0].update(source=2),
    lambda d: d['samplers'][0].update(wrapS=99),
    lambda d: d['accessors'][3].update(count=2),
    lambda d: d['materials'][0].update(alphaMode='UNKNOWN'),
    lambda d: d['materials'][0].update(alphaCutoff=float('nan')),
    lambda d: d['materials'][0]['pbrMetallicRoughness'].update(
        baseColorFactor=[1, 0, 0, 2]),
    lambda d: d['materials'][0]['pbrMetallicRoughness'].update(
        baseColorTexture=dict(index=99)),
    lambda d: d['materials'][0]['pbrMetallicRoughness'].update(
        baseColorTexture=dict(index=0, texCoord=1)),
    lambda d: d['meshes'][0]['primitives'][0]['attributes'].pop('TEXCOORD_0'),
])
def test_invalid_texture_and_material_inputs_rejected(tmp_path, modify):
    doc, binary = textured_fixture()
    modify(doc)
    with pytest.raises(ValueError):
        load(tmp_path, data=encode(doc, binary))


def test_material_factor_is_used_without_a_texture(tmp_path):
    def color(doc):
        doc['materials'] = [dict(pbrMetallicRoughness=dict(
            baseColorFactor=[0.1, 0.2, 0.3, 1]))]
        doc['meshes'][0]['primitives'][0]['material'] = 0
    item = _scene.surfaces(load(tmp_path, color))[0]
    assert item.color == (0.1, 0.2, 0.3, 1)
    assert item.texture is None

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
