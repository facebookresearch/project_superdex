# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import gc
import importlib
import pickle

from superdex.physics import loader
from test.conftest import mochi, MochiTestBase, np, np_real


class _PyBuffer(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.c_void_p),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("internal", ctypes.c_void_p),
    ]


_PYBUF_SIMPLE = 0
_PYBUF_ND = 0x0008
_PYBUF_STRIDES = 0x0010 | _PYBUF_ND
_PYBUF_F_CONTIGUOUS = 0x0040 | _PYBUF_STRIDES


class TestBindingCompatibility(MochiTestBase):
    def _get_buffer(self, value, flags):
        get_buffer = ctypes.pythonapi.PyObject_GetBuffer
        get_buffer.argtypes = [
            ctypes.py_object,
            ctypes.POINTER(_PyBuffer),
            ctypes.c_int,
        ]
        get_buffer.restype = ctypes.c_int
        view = _PyBuffer()
        self.assertEqual(0, get_buffer(value, ctypes.byref(view), flags))
        return view

    def _release_buffer(self, view):
        release_buffer = ctypes.pythonapi.PyBuffer_Release
        release_buffer.argtypes = [ctypes.POINTER(_PyBuffer)]
        release_buffer.restype = None
        release_buffer(ctypes.byref(view))

    def test_canonical_string_container_aliases(self):
        self.assertEqual("DynamicArrayReal", mochi.DynamicArrayReal.__name__)
        self.assertEqual("DynamicArrayString", mochi.DynamicArrayString.__name__)
        self.assertEqual("SpanConstString", mochi.SpanConstString.__name__)
        self.assertIs(mochi.DynamicArrayDynamicString, mochi.DynamicArrayString)
        self.assertIs(mochi.SpanConstDynamicString, mochi.SpanConstString)

    def test_dynamic_array_buffer_is_mutable_and_zero_copy(self):
        values = mochi.DynamicArrayInt([1, 2, 3])

        view = memoryview(values)

        self.assertEqual(1, view.ndim)
        self.assertEqual((3,), view.shape)
        self.assertEqual((4,), view.strides)
        self.assertFalse(view.readonly)
        view[1] = 20
        self.assertEqual(20, values[1])
        values[2] = 30
        self.assertEqual(30, view[2])

    def test_simple_buffer_request_omits_optional_metadata(self):
        values = mochi.DynamicArrayInt([1, 2, 3])
        view = self._get_buffer(values, _PYBUF_SIMPLE)
        try:
            self.assertEqual(12, view.len)
            self.assertEqual(4, view.itemsize)
            self.assertEqual(1, view.ndim)
            self.assertFalse(view.format)
            self.assertFalse(view.shape)
            self.assertFalse(view.strides)
        finally:
            self._release_buffer(view)

    def test_multidimensional_span_rejects_fortran_contiguous_request(self):
        values = mochi.DynamicArrayReal3(
            [mochi.Real3(1.0, 2.0, 3.0), mochi.Real3(4.0, 5.0, 6.0)]
        )
        span = mochi.SpanConstReal3(values)

        with self.assertRaisesRegex(BufferError, "not Fortran-contiguous"):
            self._get_buffer(span, _PYBUF_F_CONTIGUOUS)

    def test_span_buffer_keeps_backing_storage_alive(self):
        values = mochi.DynamicArrayReal([1.0, 2.0, 3.0])
        span = mochi.SpanReal(values)
        view = memoryview(span)

        self.assertEqual((3,), view.shape)
        self.assertEqual((np.dtype(np_real).itemsize,), view.strides)
        self.assertFalse(view.readonly)
        view[0] = 10.0
        self.assertEqual(10.0, values[0])

        del span
        del values
        gc.collect()
        self.assertEqual([10.0, 2.0, 3.0], list(view))

    def test_dynamic_array_numpy_asarray_aliases_buffer(self):
        values = mochi.DynamicArrayInt([1, 2, 3])

        converted = np.asarray(values)

        self.assertEqual(np.dtype(np.int32), converted.dtype)
        values[0] = 100
        self.assertEqual(100, converted[0])
        converted[1] = 200
        self.assertEqual(200, values[1])

    def test_dynamic_array_numpy_copy_is_independent(self):
        values = mochi.DynamicArrayInt([1, 2, 3])

        converted = np.array(values, copy=True)

        values[0] = 100
        self.assertEqual(1, converted[0])
        converted[1] = 200
        self.assertEqual(2, values[1])

    def test_fixed_array_numpy_conversion_is_an_independent_copy(self):
        values = mochi.Real3(1.0, 2.0, 3.0)

        converted = np.asarray(values)

        self.assertEqual(np.dtype(np_real), converted.dtype)
        values[0] = 100.0
        self.assertEqual(1.0, converted[0])
        converted[1] = 200.0
        self.assertEqual(2.0, values[1])

    def test_span_numpy_conversion_aliases_backing_storage(self):
        values = mochi.DynamicArrayReal([1.0, 2.0, 3.0])
        span = mochi.SpanReal(values)

        converted = np.asarray(span)

        self.assertEqual(np.dtype(np_real), converted.dtype)
        values[0] = 100.0
        self.assertEqual(100.0, converted[0])
        converted[1] = 200.0
        self.assertEqual(200.0, values[1])

    def test_enum_introspection_contract(self):
        rigid = mochi.ActorType.RIGID

        self.assertEqual("RIGID", rigid.name)
        self.assertIsInstance(rigid.value, int)
        self.assertIs(rigid, mochi.ActorType.__members__["RIGID"])

    def test_enum_alias_uses_standard_python_identity(self):
        canonical = mochi.ArticulatedJointType.COUNT
        alias = mochi.ArticulatedJointType.INVALID

        self.assertIs(canonical, alias)
        self.assertIs(canonical, mochi.ArticulatedJointType.__members__["INVALID"])
        self.assertIs(canonical, pickle.loads(pickle.dumps(alias)))

    def test_pickle_reducer_uses_loaded_module_name(self):
        value = mochi.Real3(1.0, 2.0, 3.0)
        constructor = value.__reduce__()[0]

        expected_module = loader.selected_native_module_name("mochi_physics")
        self.assertEqual(expected_module, constructor.__module__)
        restored = pickle.loads(pickle.dumps(value))
        self.assertIs(type(value), type(restored))
        self.assertEqual(value, restored)

    def test_shutdown_callback_has_stable_identity(self):
        expected_module = loader.selected_native_module_name("mochi_physics")
        native_module = importlib.import_module(expected_module)
        callback = native_module._mochi_shutdown_global_context_atexit
        self.assertEqual("_mochi_shutdown_global_context_atexit", callback.__name__)
        self.assertEqual(expected_module, callback.__module__)

    def test_engine_owned_pointer_identity_and_lifetime(self):
        scene = mochi.create_scene("binding_compatibility")
        actor = self._create_rigid_box_actor(scene)
        handle = actor.get_handle()

        self.assertIs(actor, scene.get_actor(handle))

        del actor
        gc.collect()
        recovered = scene.get_actor(handle)
        self.assertEqual(handle, recovered.get_handle())

        scene.destroy_actor(recovered)
        mochi.destroy_scene(scene)

    def test_invalid_span_input_raises_type_error(self):
        with self.assertRaises(TypeError):
            mochi.model.load_from_bytes(object())
