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

import copy
import math
import pickle

from test.conftest import mochi, MochiTestBase, np, np_real, without_whitespace


class TestCoreTypes(MochiTestBase):
    def test_int_array(self):
        # init default
        a = mochi.DynamicArrayInt()
        self.assertEqual(0, len(a))

        # init with size
        a = mochi.DynamicArrayInt(2)
        self.assertEqual(2, len(a))
        self.assertEqual(0, a[0])
        self.assertEqual(0, a[1])

        # init with size and value
        a = mochi.DynamicArrayInt(2, 111)
        self.assertEqual(2, len(a))
        self.assertEqual(111, a[0])
        self.assertEqual(111, a[1])

        # init with tuple
        a = mochi.DynamicArrayInt((1, 2, 3))
        self.assertEqual(3, len(a))
        self.assertEqual(1, a[0])
        self.assertEqual(2, a[1])
        self.assertEqual(3, a[2])

        # init with list
        a = mochi.DynamicArrayInt([1, 2, 3])
        self.assertEqual(3, len(a))
        self.assertEqual(1, a[0])
        self.assertEqual(2, a[1])
        self.assertEqual(3, a[2])

        # init with numpy array
        a = mochi.DynamicArrayInt(np.array([1, 2, 3], dtype=np.int32))
        self.assertEqual(3, len(a))
        self.assertEqual(1, a[0])
        self.assertEqual(2, a[1])
        self.assertEqual(3, a[2])

        # bool conversion
        a = mochi.DynamicArrayInt()
        self.assertFalse(a)
        a.append(0)
        self.assertTrue(a)

        # get/set by index
        a = mochi.DynamicArrayInt(2)
        a[0] = 11
        a[1] = 22
        self.assertEqual(11, a[0])
        self.assertEqual(22, a[1])

        # iterate
        a = mochi.DynamicArrayInt([1, 2, 3])
        i = 0
        for item in a:
            self.assertEqual(i + 1, item)
            i += 1

        # append
        a = mochi.DynamicArrayInt()
        a.append(123)
        a.append(456)
        self.assertEqual(2, len(a))
        self.assertEqual(123, a[0])
        self.assertEqual(456, a[1])

        # extend
        a = mochi.DynamicArrayInt([1, 2])
        a.extend(mochi.DynamicArrayInt([3]))
        a.extend([4])
        self.assertEqual(4, len(a))
        self.assertEqual(1, a[0])
        self.assertEqual(2, a[1])
        self.assertEqual(3, a[2])
        self.assertEqual(4, a[3])

        # clear
        a = mochi.DynamicArrayInt([1, 2, 3])
        a.clear()
        self.assertEqual(0, len(a))

        # empty
        a = mochi.DynamicArrayInt([1, 2, 3])
        self.assertFalse(a.empty())
        a.clear()
        self.assertTrue(a.empty())

        # size
        a = mochi.DynamicArrayInt([1, 2, 3])
        self.assertEqual(3, a.size())

        # reserve
        a = mochi.DynamicArrayInt()
        a.reserve(1)
        self.assertEqual(0, a.size())
        self.assertEqual(1, a.capacity())

        # resize
        a = mochi.DynamicArrayInt()
        a.resize(1)
        self.assertEqual(1, a.size())
        self.assertEqual(0, a[0])
        a.resize(2, 123)
        self.assertEqual(2, a.size())
        self.assertEqual(0, a[0])
        self.assertEqual(123, a[1])
        a.resize(1)
        self.assertEqual(1, a.size())
        a.resize(0)
        self.assertEqual(0, a.size())

        # Conversion to numpy.array
        a = mochi.DynamicArrayInt([1, 2, 3])
        npa = np.array(a)
        self.assertEqual(3, len(npa))
        self.assertEqual(1, npa[0])
        self.assertEqual(2, npa[1])
        self.assertEqual(3, npa[2])
        self.assertEqual(np.int32, npa.dtype)
        a = mochi.DynamicArrayInt([4, 5])
        npa = np.array(a, dtype=np.int64)
        self.assertEqual(2, len(npa))
        self.assertEqual(4, npa[0])
        self.assertEqual(5, npa[1])
        self.assertEqual(np.int64, npa.dtype)

        # Conversion to list
        a = mochi.DynamicArrayInt([1, 2, 3])
        self.assertEqual([1, 2, 3], a.tolist())

        # Conversion to string
        a = mochi.DynamicArrayInt([1, 2, 3])
        self.assertEqual("[1,2,3]", without_whitespace(str(a)))

        # Equality
        self.assertEqual(mochi.DynamicArrayInt(), mochi.DynamicArrayInt())
        self.assertEqual(
            mochi.DynamicArrayInt([1, 2, 3]), mochi.DynamicArrayInt([1, 2, 3])
        )
        self.assertNotEqual(mochi.DynamicArrayInt([1, 2, 3]), mochi.DynamicArrayInt())
        self.assertNotEqual(
            mochi.DynamicArrayInt([1, 2, 3]), mochi.DynamicArrayInt([1, 2])
        )
        self.assertNotEqual(
            mochi.DynamicArrayInt([1, 2, 3]), mochi.DynamicArrayInt([1, 2, 4])
        )

        # pickle serialization
        a = mochi.DynamicArrayInt([1, 2, 3])
        serialized_data = pickle.dumps(a)
        a2 = pickle.loads(serialized_data)
        self.assertEqual(a, a2)

        # copy.copy / copy.deepcopy: independent copies via the C++ copy ctor.
        a = mochi.DynamicArrayInt([1, 2, 3])
        for c in (copy.copy(a), copy.deepcopy(a)):
            self.assertIsInstance(c, mochi.DynamicArrayInt)
            self.assertEqual(a, c)
            a[0] = 99
            self.assertNotEqual(a, c)
            a[0] = 1  # restore for next iteration

    def test_real2(self):
        # Default constructor
        v = mochi.Real2()
        self.assertEqual(v[0], 0)
        self.assertEqual(v[1], 0)

        # Construct with values
        v = mochi.Real2(1, 2)
        self.assertEqual(v[0], 1)
        self.assertEqual(v[1], 2)
        self.assertNotEqual(v, mochi.Real2())

        # Equivalent contructor syntax
        self.assertEqual(v, mochi.Real2([1, 2]))
        self.assertEqual(v, mochi.Real2(np.array([1, 2], dtype=np_real)))

        # Array access
        v[0] = 4
        v[1] = 5
        self.assertEqual(v, mochi.Real2([4, 5]))

        # Length
        self.assertEqual(2, len(v))

        # String conversion
        self.assertEqual("[4,5]", without_whitespace(str(v)))

        # List conversion
        self.assertEqual([4, 5], v.tolist())

        # numpy.array conversion
        npa = np.array(v, dtype=np.float32)
        self.assertEqual(2, len(npa))
        self.assertEqual(4, npa[0])
        self.assertEqual(5, npa[1])
        self.assertEqual(np.float32, npa.dtype)
        npa = np.array(v, dtype=np.float64)
        self.assertEqual(2, len(npa))
        self.assertEqual(4, npa[0])
        self.assertEqual(5, npa[1])
        self.assertEqual(np.float64, npa.dtype)

        # Math operators
        a = mochi.Real2(1, 2)
        b = mochi.Real2(4, 5)
        self.assertEqual(mochi.Real2(5, 7), a + b)
        self.assertEqual(mochi.Real2(-3, -3), a - b)
        self.assertEqual(mochi.Real2(4, 10), a * b)
        self.assertEqual(mochi.Real2(1 / 4, 2 / 5), a / b)
        self.assertEqual(mochi.Real2(2, 3), a + 1)
        self.assertEqual(mochi.Real2(0, 1), a - 1)
        self.assertEqual(mochi.Real2(2, 4), a * 2)
        self.assertEqual(mochi.Real2(2, 4), a / 0.5)
        self.assertEqual(mochi.Real2(2, 3), 1 + a)
        self.assertEqual(mochi.Real2(0, -1), 1 - a)
        self.assertEqual(mochi.Real2(2, 4), 2 * a)
        self.assertEqual(mochi.Real2(2, 1), 2 / a)
        self.assertEqual(mochi.Real2(-1, -2), -a)

        # Math assignment operators
        a = mochi.Real2(1, 2)
        a += a
        self.assertEqual(mochi.Real2(2, 4), a)
        a -= a
        self.assertEqual(mochi.Real2(0, 0), a)
        a = mochi.Real2(1, 2)
        a *= a
        self.assertEqual(mochi.Real2(1, 4), a)
        a /= a
        self.assertEqual(mochi.Real2(1, 1), a)
        a = mochi.Real2(1, 2)
        a += 1
        self.assertEqual(mochi.Real2(2, 3), a)
        a -= 1
        self.assertEqual(mochi.Real2(1, 2), a)
        a *= 2
        self.assertEqual(mochi.Real2(2, 4), a)
        a /= 2
        self.assertEqual(mochi.Real2(1, 2), a)

        # normalize
        self.assertEqual(mochi.Real2(1, 0), mochi.normalize(mochi.Real2(2, 0)))

        # pickle serialization
        a = mochi.Real2(1.1, 2.2)
        serialized_data = pickle.dumps(a)
        a2 = pickle.loads(serialized_data)
        self.assertEqual(a, a2)  # Exact equality

        # pickle a DynamicArray of Real2
        a = mochi.DynamicArrayReal2([mochi.Real2(1, 2), mochi.Real2(3, 4)])
        serialized_data = pickle.dumps(a)
        a2 = pickle.loads(serialized_data)
        self.assertEqual(a, a2)  # Exact equality

    def test_real3(self):
        # Default constructor
        v = mochi.Real3()
        self.assertEqual(v[0], 0)
        self.assertEqual(v[1], 0)
        self.assertEqual(v[2], 0)

        # Construct with values
        v = mochi.Real3(1, 2, 3)
        self.assertEqual(v[0], 1)
        self.assertEqual(v[1], 2)
        self.assertEqual(v[2], 3)
        self.assertNotEqual(v, mochi.Real3())

        # Equivalent contructor syntax
        self.assertEqual(v, mochi.Real3([1, 2, 3]))
        self.assertEqual(v, mochi.Real3(np.array([1, 2, 3], dtype=np_real)))

        # Array access
        v[0] = 4
        v[1] = 5
        v[2] = 6
        self.assertEqual(v, mochi.Real3([4, 5, 6]))

        # Length
        self.assertEqual(3, len(v))

        # Conversion to string
        self.assertEqual("[4,5,6]", without_whitespace(str(v)))

        # Conversion to list
        self.assertEqual([4, 5, 6], v.tolist())

        # Conversion to numpy.array
        npa = np.array(v, dtype=np.float32)
        self.assertEqual(3, len(npa))
        self.assertEqual(4, npa[0])
        self.assertEqual(5, npa[1])
        self.assertEqual(6, npa[2])
        self.assertEqual(np.float32, npa.dtype)
        npa = np.array(v, dtype=np.float64)
        self.assertEqual(3, len(npa))
        self.assertEqual(4, npa[0])
        self.assertEqual(5, npa[1])
        self.assertEqual(6, npa[2])
        self.assertEqual(np.float64, npa.dtype)

        # Math operators
        a = mochi.Real3(1, 2, 3)
        b = mochi.Real3(4, 5, 6)
        self.assertEqual(mochi.Real3(5, 7, 9), a + b)
        self.assertEqual(mochi.Real3(-3, -3, -3), a - b)
        self.assertEqual(mochi.Real3(4, 10, 18), a * b)
        self.assertEqual(mochi.Real3(1 / 4, 2 / 5, 3 / 6), a / b)
        self.assertEqual(mochi.Real3(2, 3, 4), a + 1)
        self.assertEqual(mochi.Real3(0, 1, 2), a - 1)
        self.assertEqual(mochi.Real3(2, 4, 6), a * 2)
        self.assertEqual(mochi.Real3(2, 4, 6), a / 0.5)
        self.assertEqual(mochi.Real3(2, 3, 4), 1 + a)
        self.assertEqual(mochi.Real3(0, -1, -2), 1 - a)
        self.assertEqual(mochi.Real3(2, 4, 6), 2 * a)
        self.assertEqual(mochi.Real3(2, 1, 2 / 3), 2 / a)
        self.assertEqual(mochi.Real3(-1, -2, -3), -a)

        # Math assignment operators
        a = mochi.Real3(1, 2, 3)
        a += a
        self.assertEqual(mochi.Real3(2, 4, 6), a)
        a -= a
        self.assertEqual(mochi.Real3(0, 0, 0), a)
        a = mochi.Real3(1, 2, 3)
        a *= a
        self.assertEqual(mochi.Real3(1, 4, 9), a)
        a /= a
        self.assertEqual(mochi.Real3(1, 1, 1), a)
        a = mochi.Real3(1, 2, 3)
        a += 1
        self.assertEqual(mochi.Real3(2, 3, 4), a)
        a -= 1
        self.assertEqual(mochi.Real3(1, 2, 3), a)
        a *= 2
        self.assertEqual(mochi.Real3(2, 4, 6), a)
        a /= 2
        self.assertEqual(mochi.Real3(1, 2, 3), a)

        # normalize
        self.assertEqual(mochi.Real3(1, 0, 0), mochi.normalize(mochi.Real3(2, 0, 0)))

        # pickle serialization
        a = mochi.Real3(1.1, 2.2, 3.3)
        serialized_data = pickle.dumps(a)
        a2 = pickle.loads(serialized_data)
        self.assertEqual(a, a2)  # Exact equality

        # pickle a DynamicArray of Real3
        a = mochi.DynamicArrayReal3([mochi.Real3(1, 2, 3), mochi.Real3(3, 4, 5)])
        serialized_data = pickle.dumps(a)
        a2 = pickle.loads(serialized_data)
        self.assertEqual(a, a2)  # Exact equality

        # copy.copy / copy.deepcopy: NdArray stores its elements inline, so the C++
        # copy ctor produces an independent copy.
        v = mochi.Real3(1, 2, 3)
        for c in (copy.copy(v), copy.deepcopy(v)):
            self.assertIsInstance(c, mochi.Real3)
            self.assertEqual(v, c)
            v[0] = 99
            self.assertNotEqual(v, c)
            v[0] = 1  # restore for next iteration

    def test_quaternion(self):
        # Default constructor (identity quaternion)
        q = mochi.Quaternion()
        self.assertEqual(q[0], 0)  # x
        self.assertEqual(q[1], 0)  # y
        self.assertEqual(q[2], 0)  # z
        self.assertEqual(q[3], 1)  # w

        # Construct with values (x, y, z, w)
        q = mochi.Quaternion(0.1, 0.2, 0.3, 0.4)
        self.assertAlmostEqual(q[0], 0.1)
        self.assertAlmostEqual(q[1], 0.2)
        self.assertAlmostEqual(q[2], 0.3)
        self.assertAlmostEqual(q[3], 0.4)

        # Equivalent constructor syntax
        self.assertEqual(q, mochi.Quaternion([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(
            q, mochi.Quaternion(np.array([0.1, 0.2, 0.3, 0.4], dtype=np_real))
        )

        # Array access
        q[0] = 0.5
        q[1] = 0.6
        q[2] = 0.7
        q[3] = 0.8
        self.assertEqual(q, mochi.Quaternion([0.5, 0.6, 0.7, 0.8]))

        # Length
        self.assertEqual(4, len(q))

        # String conversion
        self.assertEqual(
            "[1,2,3,4]", without_whitespace(str(mochi.Quaternion(1, 2, 3, 4)))
        )

        # tolist
        self.assertEqual([1, 2, 3, 4], mochi.Quaternion(1, 2, 3, 4).tolist())

        # identity
        identity = mochi.Quaternion.identity()
        self.assertEqual(identity, mochi.Quaternion(0, 0, 0, 1))

        # zero
        zero = mochi.Quaternion.zero()
        self.assertEqual(zero, mochi.Quaternion(0, 0, 0, 0))

        # from_axis_angle
        angle = math.pi / 2
        q_axis_angle = mochi.Quaternion.from_axis_angle([0, 0, 1], angle)
        self.assertAlmostEqual(q_axis_angle[0], 0)  # x
        self.assertAlmostEqual(q_axis_angle[1], 0)  # y
        self.assertAlmostEqual(q_axis_angle[2], math.sin(angle / 2))  # z
        self.assertAlmostEqual(q_axis_angle[3], math.cos(angle / 2))  # w

        # from_rotation_vector
        q_rot_vec = mochi.Quaternion.from_rotation_vector([0, 0, angle])
        self.assertAlmostEqual(q_rot_vec[0], q_axis_angle[0])
        self.assertAlmostEqual(q_rot_vec[1], q_axis_angle[1])
        self.assertAlmostEqual(q_rot_vec[2], q_axis_angle[2])
        self.assertAlmostEqual(q_rot_vec[3], q_axis_angle[3])

        # rotation_x
        rot_x = mochi.Quaternion.rotation_x(angle)
        self.assertAlmostEqual(rot_x[0], math.sin(angle / 2))
        self.assertAlmostEqual(rot_x[1], 0)
        self.assertAlmostEqual(rot_x[2], 0)
        self.assertAlmostEqual(rot_x[3], math.cos(angle / 2))

        # rotation_y
        rot_y = mochi.Quaternion.rotation_y(angle)
        self.assertAlmostEqual(rot_y[0], 0)
        self.assertAlmostEqual(rot_y[1], math.sin(angle / 2))
        self.assertAlmostEqual(rot_y[2], 0)
        self.assertAlmostEqual(rot_y[3], math.cos(angle / 2))

        # rotation_z
        rot_z = mochi.Quaternion.rotation_z(angle)
        self.assertAlmostEqual(rot_z[0], 0)
        self.assertAlmostEqual(rot_z[1], 0)
        self.assertAlmostEqual(rot_z[2], math.sin(angle / 2))
        self.assertAlmostEqual(rot_z[3], math.cos(angle / 2))

        # Test quaternion arithmetic operators
        q1 = mochi.Quaternion(0.1, 0.2, 0.3, 0.4)
        q2 = mochi.Quaternion(0.5, 0.6, 0.7, 0.8)

        # Addition
        q_add = q1 + q2
        self.assertAlmostEqual(q_add[0], 0.6)
        self.assertAlmostEqual(q_add[1], 0.8)
        self.assertAlmostEqual(q_add[2], 1.0)
        self.assertAlmostEqual(q_add[3], 1.2)

        # Subtraction
        q_sub = q2 - q1
        self.assertAlmostEqual(q_sub[0], 0.4)
        self.assertAlmostEqual(q_sub[1], 0.4)
        self.assertAlmostEqual(q_sub[2], 0.4)
        self.assertAlmostEqual(q_sub[3], 0.4)

        # Unary negation
        q_neg = -q1
        self.assertAlmostEqual(q_neg[0], -0.1)
        self.assertAlmostEqual(q_neg[1], -0.2)
        self.assertAlmostEqual(q_neg[2], -0.3)
        self.assertAlmostEqual(q_neg[3], -0.4)

        # Scalar multiplication
        q_scaled = q1 * 2.0
        self.assertAlmostEqual(q_scaled[0], 0.2)
        self.assertAlmostEqual(q_scaled[1], 0.4)
        self.assertAlmostEqual(q_scaled[2], 0.6)
        self.assertAlmostEqual(q_scaled[3], 0.8)

        # Scalar division
        q_divided = q_scaled / 2.0
        self.assertAlmostEqual(q_divided[0], 0.1)
        self.assertAlmostEqual(q_divided[1], 0.2)
        self.assertAlmostEqual(q_divided[2], 0.3)
        self.assertAlmostEqual(q_divided[3], 0.4)

        # Test equality operators
        q_copy = mochi.Quaternion(0.1, 0.2, 0.3, 0.4)
        self.assertTrue(q1 == q_copy)
        self.assertFalse(q1 != q_copy)
        self.assertFalse(q1 == q2)
        self.assertTrue(q1 != q2)

        # normalize
        self.assertEqual(
            mochi.Quaternion(0, 0, 0, 1), mochi.normalize(mochi.Quaternion(0, 0, 0, 2))
        )

        # pickle serialization
        q = mochi.Quaternion(1.1, 2.2, 3.3, 4.4)
        serialized_data = pickle.dumps(q)
        q2 = pickle.loads(serialized_data)
        self.assertEqual(q, q2)  # Exact equality

        # copy.copy / copy.deepcopy: independent copies via the C++ copy ctor.
        q = mochi.Quaternion(1, 2, 3, 4)
        for c in (copy.copy(q), copy.deepcopy(q)):
            self.assertIsInstance(c, mochi.Quaternion)
            self.assertEqual(q, c)
            q[0] = 99
            self.assertNotEqual(q, c)
            q[0] = 1  # restore for next iteration

    def test_transform_rt(self):
        # Default constructor (identity transform)
        t = mochi.TransformRT()
        self.assertEqual(t.rotation, mochi.Quaternion.identity())
        self.assertEqual(t.translation, mochi.Real3(0, 0, 0))

        # Constructor with rotation only
        rotation = mochi.Quaternion.rotation_z(math.pi / 2)
        t_rot = mochi.TransformRT(rotation)
        self.assertEqual(t_rot.rotation, rotation)
        self.assertEqual(t_rot.translation, mochi.Real3(0, 0, 0))
        self.assertEqual(
            t_rot, mochi.TransformRT(rotation=rotation)
        )  # keyword argument this time
        with self.assertRaises(TypeError):
            mochi.TransformRT(rotation=[1, 2, 3])  # Wrong length for Quaternion

        # 4-element positional argument now creates rotation via implicit conversion
        t_rot_implicit = mochi.TransformRT([0.5, 0.5, 0.5, 0.5])
        self.assertEqual(t_rot_implicit.rotation, mochi.Quaternion(0.5, 0.5, 0.5, 0.5))
        self.assertEqual(t_rot_implicit.translation, mochi.Real3(0, 0, 0))

        # Constructor with translation only
        t_trans = mochi.TransformRT([1, 2, 3])
        self.assertEqual(t_trans.rotation, mochi.Quaternion.identity())
        self.assertEqual(t_trans.translation, mochi.Real3(1, 2, 3))
        self.assertEqual(
            t_trans, mochi.TransformRT(translation=[1, 2, 3])
        )  # keyword argument this time

        # Constructor with both rotation and translation
        t_both = mochi.TransformRT(rotation, [1, 2, 3])
        self.assertEqual(t_both.rotation, rotation)
        self.assertEqual(t_both.translation, mochi.Real3(1, 2, 3))
        self.assertEqual(
            t_both, mochi.TransformRT(rotation=rotation, translation=[1, 2, 3])
        )  # keyword argument this time

        # Static identity method
        identity = mochi.TransformRT.identity()
        self.assertEqual(identity.rotation, mochi.Quaternion.identity())
        self.assertEqual(identity.translation, mochi.Real3(0, 0, 0))

        # Test property setters
        t = mochi.TransformRT()
        new_rotation = mochi.normalize(mochi.Quaternion([1, 2, 3, 4]))
        t = mochi.TransformRT()
        t.rotation = new_rotation
        t.translation = [5, 6, 7]
        self.assertEqual(t.rotation, new_rotation)
        self.assertEqual(t.translation, mochi.Real3(5, 6, 7))

        # Test string representation
        mochi.set_log_callback(
            lambda channel, message, file, line: ()
        )  # Suppresss warning
        try:
            t = mochi.TransformRT(rotation=[1, 2, 3, 4], translation=[5, 6, 7])
            self.assertEqual(
                '{"rotation":[1,2,3,4],"translation":[5,6,7]}',
                without_whitespace(str(t)),
            )
        finally:
            mochi.set_log_callback(None)  # Restore default logging

        # Test equality operators
        t1 = mochi.TransformRT(rotation=[0, 0, 0, 1], translation=[1, 2, 3])
        t2 = mochi.TransformRT(rotation=[0, 0, 0, 1], translation=[1, 2, 3])
        t3 = mochi.TransformRT(
            rotation=mochi.normalize(mochi.Quaternion(0.1, 0, 0, 1)),
            translation=[1, 2, 3],
        )
        self.assertTrue(t1 == t2)
        self.assertFalse(t1 != t2)
        self.assertFalse(t1 == t3)
        self.assertTrue(t1 != t3)

        # Test multiplication operator (transform composition)
        # Create two transforms: one rotation around Z, one translation
        rot_z = mochi.TransformRT(rotation=mochi.Quaternion.rotation_z(math.pi / 2))
        trans_x = mochi.TransformRT(translation=[1, 0, 0])
        composed = rot_z * trans_x
        self.assertEqual(composed.rotation, rot_z.rotation)
        self.assertAlmostEqual(composed.translation[0], 0, places=6)
        self.assertAlmostEqual(composed.translation[1], 1, places=6)
        self.assertAlmostEqual(composed.translation[2], 0, places=6)

        # Test multiplication assignment operator
        t1 = mochi.TransformRT(
            rotation=mochi.Quaternion.rotation_x(math.pi / 4), translation=[1, 0, 0]
        )
        t2 = mochi.TransformRT(
            rotation=mochi.Quaternion.rotation_y(math.pi / 4), translation=[0, 1, 0]
        )
        original_t1 = mochi.TransformRT(
            rotation=t1.rotation, translation=t1.translation
        )
        t1 *= t2

        # Should be equivalent to t1 = original_t1 * t2
        expected = original_t1 * t2
        self.assertEqual(t1.rotation, expected.rotation)
        self.assertEqual(t1.translation, expected.translation)

        # Test identity transform properties
        identity = mochi.TransformRT.identity()

        # Test chaining multiple transforms
        t1 = mochi.TransformRT(
            rotation=mochi.Quaternion.rotation_x(math.pi / 6), translation=[1, 0, 0]
        )
        t2 = mochi.TransformRT(
            rotation=mochi.Quaternion.rotation_y(math.pi / 6), translation=[0, 1, 0]
        )
        t3 = mochi.TransformRT(
            rotation=mochi.Quaternion.rotation_z(math.pi / 6), translation=[0, 0, 1]
        )

        # Test associativity: (t1 * t2) * t3 == t1 * (t2 * t3)
        left_assoc = (t1 * t2) * t3
        right_assoc = t1 * (t2 * t3)

        # Due to floating point precision, we need to use approximate equality
        for i in range(4):
            self.assertAlmostEqual(
                left_assoc.rotation[i], right_assoc.rotation[i], places=6
            )
        for i in range(3):
            self.assertAlmostEqual(
                left_assoc.translation[i], right_assoc.translation[i], places=6
            )

        # pickle serialization
        t = mochi.TransformRT(
            rotation=mochi.normalize(mochi.Quaternion(1, 2, 3, 4)),
            translation=[0.1, 0.2, 0.3],
        )
        serialized_data = pickle.dumps(t)
        t2 = pickle.loads(serialized_data)
        self.assertEqual(t, t2)  # Exact equality

        # copy.copy / copy.deepcopy: independent copies via the C++ copy ctor.
        t = mochi.TransformRT(
            rotation=mochi.Quaternion(0, 0, 0, 1),
            translation=mochi.Real3(1, 2, 3),
        )
        for c in (copy.copy(t), copy.deepcopy(t)):
            self.assertIsInstance(c, mochi.TransformRT)
            self.assertEqual(t, c)
            t.translation = mochi.Real3(9, 9, 9)
            self.assertNotEqual(t, c)
            t.translation = mochi.Real3(1, 2, 3)  # restore for next iteration

    def test_transform_rt_inverse(self):
        # Test inverse of identity transform
        identity = mochi.TransformRT.identity()
        identity_inv = identity.inverse()
        self.assertEqual(identity_inv.rotation, mochi.Quaternion.identity())
        self.assertEqual(identity_inv.translation, mochi.Real3(0, 0, 0))

        # Test inverse of translation only
        t_trans = mochi.TransformRT(translation=[1, 2, 3])
        t_trans_inv = t_trans.inverse()
        self.assertEqual(t_trans_inv.rotation, mochi.Quaternion.identity())
        self.assertAlmostEqual(t_trans_inv.translation[0], -1, places=6)
        self.assertAlmostEqual(t_trans_inv.translation[1], -2, places=6)
        self.assertAlmostEqual(t_trans_inv.translation[2], -3, places=6)

        # Test inverse of rotation only
        rotation = mochi.Quaternion.rotation_z(math.pi / 2)
        t_rot = mochi.TransformRT(rotation=rotation)
        t_rot_inv = t_rot.inverse()
        # Rotation inverse is the conjugate
        expected_inv_rot = mochi.Quaternion.rotation_z(-math.pi / 2)
        for i in range(4):
            self.assertAlmostEqual(t_rot_inv.rotation[i], expected_inv_rot[i], places=6)

        # Test inverse of combined rotation and translation
        # For a general transform, t * t.inverse() should be identity
        t_combined = mochi.TransformRT(
            rotation=mochi.Quaternion.rotation_x(math.pi / 4),
            translation=[1, 2, 3],
        )
        t_combined_inv = t_combined.inverse()
        result = t_combined * t_combined_inv

        # Result should be identity (within floating point tolerance)
        self.assertAlmostEqual(result.rotation[0], 0, places=6)
        self.assertAlmostEqual(result.rotation[1], 0, places=6)
        self.assertAlmostEqual(result.rotation[2], 0, places=6)
        self.assertAlmostEqual(result.rotation[3], 1, places=6)
        self.assertAlmostEqual(result.translation[0], 0, places=6)
        self.assertAlmostEqual(result.translation[1], 0, places=6)
        self.assertAlmostEqual(result.translation[2], 0, places=6)

    def test_aabb(self):
        # Default constructor (should create empty/zero AABB)
        aabb = mochi.Aabb()
        self.assertEqual(aabb.min, mochi.Real3(0, 0, 0))
        self.assertEqual(aabb.max, mochi.Real3(0, 0, 0))

        # Constructor with min and max
        aabb = mochi.Aabb([-1, -2, -3], [4, 5, 6])
        self.assertEqual(aabb.min, mochi.Real3(-1, -2, -3))
        self.assertEqual(aabb.max, mochi.Real3(4, 5, 6))

        # Test property getters
        self.assertEqual(aabb.min, mochi.Real3(-1, -2, -3))
        self.assertEqual(aabb.max, mochi.Real3(4, 5, 6))

        # Test property setters
        aabb.min = [-5, -6, -7]
        aabb.max = [8, 9, 10]
        self.assertEqual(aabb.min, mochi.Real3(-5, -6, -7))
        self.assertEqual(aabb.max, mochi.Real3(8, 9, 10))

        # Test get_center method
        aabb = mochi.Aabb(min=[-2, -4, -6], max=[4, 8, 12])
        center = aabb.get_center()
        expected_center = mochi.Real3(1, 2, 3)  # (-2+4)/2, (-4+8)/2, (-6+12)/2
        self.assertEqual(center, expected_center)

        # Test get_size method
        size = aabb.get_size()
        expected_size = mochi.Real3(6, 12, 18)  # 4-(-2), 8-(-4), 12-(-6)
        self.assertEqual(size, expected_size)

        # Test with unit cube
        unit_aabb = mochi.Aabb([0, 0, 0], [1, 1, 1])
        self.assertEqual(unit_aabb.get_center(), mochi.Real3(0.5, 0.5, 0.5))
        self.assertEqual(unit_aabb.get_size(), mochi.Real3(1, 1, 1))

        # Test with negative coordinates
        neg_aabb = mochi.Aabb([-10, -20, -30], [-5, -10, -15])
        self.assertEqual(neg_aabb.get_center(), mochi.Real3(-7.5, -15, -22.5))
        self.assertEqual(neg_aabb.get_size(), mochi.Real3(5, 10, 15))

        # Test equality operators
        aabb1 = mochi.Aabb(min=[1, 2, 3], max=[4, 5, 6])
        aabb2 = mochi.Aabb(min=[1, 2, 3], max=[4, 5, 6])
        aabb3 = mochi.Aabb(min=[1, 2, 3], max=[4, 5, 7])

        self.assertTrue(aabb1 == aabb2)
        self.assertFalse(aabb1 != aabb2)
        self.assertFalse(aabb1 == aabb3)
        self.assertTrue(aabb1 != aabb3)

        # Test string representation
        aabb = mochi.Aabb([1, 2, 3], [4, 5, 6])
        self.assertEqual('{"max":[4,5,6],"min":[1,2,3]}', without_whitespace(str(aabb)))

        # pickle serialization
        aabb = mochi.Aabb([1, 2, 3], [4, 5, 6])
        serialized_data = pickle.dumps(aabb)
        aabb2 = pickle.loads(serialized_data)
        self.assertEqual(aabb, aabb2)  # Exact equality

        # copy.copy / copy.deepcopy: independent copies via the C++ copy ctor.
        aabb = mochi.Aabb([1, 2, 3], [4, 5, 6])
        for c in (copy.copy(aabb), copy.deepcopy(aabb)):
            self.assertIsInstance(c, mochi.Aabb)
            self.assertEqual(aabb, c)
            aabb.max = mochi.Real3(9, 9, 9)
            self.assertNotEqual(aabb, c)
            aabb.max = mochi.Real3(4, 5, 6)  # restore for next iteration

    def test_coordinate_space(self):
        # Default constructor matches the Default() factory.
        cs = mochi.CoordinateSpace()
        self.assertEqual(cs, mochi.CoordinateSpace.default())
        self.assertEqual(cs.axes, mochi.CoordinateSpaceAxes.FLU)
        self.assertEqual(cs.axes, mochi.CoordinateSpaceAxes.DEFAULT)
        self.assertEqual(cs.units_per_meter, 1)
        cs.validate()

        # Other conventions.
        unreal = mochi.CoordinateSpace.unreal()
        self.assertEqual(unreal.axes, mochi.CoordinateSpaceAxes.FRU)
        self.assertEqual(unreal.units_per_meter, 100)
        self.assertNotEqual(unreal, cs)

        # Fields are settable, by keyword too.
        cs.axes = mochi.CoordinateSpaceAxes.FRU
        cs.units_per_meter = 100
        self.assertEqual(
            cs,
            mochi.CoordinateSpace(
                axes=mochi.CoordinateSpaceAxes.FRU, units_per_meter=100
            ),
        )

        # Validation rejects a non-positive scale.
        with self.assertRaises(mochi.Error):
            mochi.CoordinateSpace(units_per_meter=0).validate()

    def test_shape_handle(self):
        a = mochi.ShapeHandle()
        self.assertEqual(0, a.value)
        self.assertFalse(a.is_valid())
        b = mochi.ShapeHandle()
        b.value = 123
        self.assertTrue(a == a)
        self.assertTrue(a != b)
        self.assertFalse(a != a)
        self.assertFalse(a == b)

    def test_actor_handle(self):
        a = mochi.ActorHandle()
        self.assertEqual(0, a.value)
        self.assertFalse(a.is_valid())
        b = mochi.ActorHandle()
        b.value = 123
        self.assertTrue(a == a)
        self.assertTrue(a != b)
        self.assertFalse(a != a)
        self.assertFalse(a == b)

    def test_handle_sortability_and_hashing(self):
        def test_handles(handles):
            sorted_handles = sorted(handles)
            for a, b in zip(sorted_handles[:-1], sorted_handles[1:]):
                self.assertLess(a, b)
            hashes = [hash(h) for h in handles]
            for i in range(0, len(hashes)):
                for j in range(i + 1, len(hashes)):
                    self.assertNotEqual(hashes[i], hashes[j])

        # Test scene handles.
        scene_1 = mochi.create_scene("")
        scene_2 = mochi.create_scene("")
        test_handles([scene_1.get_handle(), scene_2.get_handle()])

        # Test actor handles.
        actor_1 = self._create_rigid_box_actor(scene_1)
        actor_2 = self._create_rigid_box_actor(scene_1)
        actor_3 = self._create_rigid_box_actor(scene_1)
        test_handles([actor_1.get_handle(), actor_2.get_handle(), actor_3.get_handle()])

        mochi.destroy_scene(scene_1)
        mochi.destroy_scene(scene_2)

    def test_dynamic_array_getitem_returns_reference(self):
        """Verify __getitem__ and __iter__ behavior for DynamicArray.

        For struct types: returns a mutable reference (Python list semantics).
        For arithmetic types: returns a value copy (Python int/float are immutable).
        """
        # --- Struct type (TransformRT): __getitem__ returns a mutable reference ---
        arr = mochi.DynamicArrayTransformRT([mochi.TransformRT(), mochi.TransformRT()])

        # Modify element via __getitem__ — change persists
        arr[0].translation = mochi.Real3(1, 2, 3)
        self.assertEqual(arr[0].translation, mochi.Real3(1, 2, 3))

        # Modify elements via __iter__ — changes persist
        for t in arr:
            t.translation = mochi.Real3(7, 8, 9)
        self.assertEqual(arr[0].translation, mochi.Real3(7, 8, 9))
        self.assertEqual(arr[1].translation, mochi.Real3(7, 8, 9))

        # __setitem__ still works
        arr[1] = mochi.TransformRT(mochi.Quaternion.identity(), mochi.Real3(4, 5, 6))
        self.assertEqual(arr[1].translation, mochi.Real3(4, 5, 6))

        # --- Arithmetic type (int): __getitem__ returns a value copy ---
        int_arr = mochi.DynamicArrayInt([10, 20, 30])

        # Reading works as expected
        self.assertEqual(int_arr[0], 10)
        self.assertEqual(int_arr[1], 20)

        # Modifying a read value does NOT affect the array (Python int is immutable)
        x = int_arr[0]  # noqa: F841
        x = 999  # noqa: F841  Rebinds local; original array unchanged
        self.assertEqual(int_arr[0], 10)  # Unchanged

        # Iteration yields copies (Python int is immutable)
        for val in int_arr:
            val = 0  # noqa: F841  Rebinds local variable, doesn't affect array
        self.assertEqual(int_arr[0], 10)  # Unchanged

        # Must use __setitem__ to modify arithmetic arrays
        int_arr[0] = 42
        self.assertEqual(int_arr[0], 42)

    def test_span_getitem_returns_reference(self):
        """Verify __getitem__ and __iter__ behavior for Span.

        For mutable spans of structs: returns a mutable reference (modifications
        through the span are visible in the underlying DynamicArray).
        For arithmetic spans: returns a value copy.
        Const spans block __setitem__.
        """
        # --- Mutable span of structs (SpanTransformRT) ---
        arr = mochi.DynamicArrayTransformRT([mochi.TransformRT(), mochi.TransformRT()])
        span = mochi.SpanTransformRT(arr)
        self.assertEqual(2, len(span))

        # __getitem__ returns a mutable reference — modification persists
        span[0].translation = mochi.Real3(1, 2, 3)
        self.assertEqual(arr[0].translation, mochi.Real3(1, 2, 3))

        # __iter__ yields mutable references — modifications persist
        for t in span:
            t.translation = mochi.Real3(7, 8, 9)
        self.assertEqual(arr[0].translation, mochi.Real3(7, 8, 9))
        self.assertEqual(arr[1].translation, mochi.Real3(7, 8, 9))

        # __setitem__ works on mutable spans
        span[1] = mochi.TransformRT(mochi.Quaternion.identity(), mochi.Real3(4, 5, 6))
        self.assertEqual(arr[1].translation, mochi.Real3(4, 5, 6))

        # --- Const span of structs (SpanConstTransformRT) ---
        const_span = mochi.SpanConstTransformRT(arr)
        self.assertEqual(2, len(const_span))

        # __getitem__ returns correct type (readable)
        t = const_span[0]
        self.assertEqual(type(t), mochi.TransformRT)
        self.assertEqual(t.translation, mochi.Real3(7, 8, 9))

        # __setitem__ is blocked on const spans
        with self.assertRaises(TypeError):
            const_span[0] = mochi.TransformRT()

        # __getitem__ returns a copy — field mutation does not affect underlying data
        original_translation = arr[0].translation
        const_span[0].translation = mochi.Real3(99.0, 99.0, 99.0)
        self.assertEqual(arr[0].translation, original_translation)

        # __iter__ yields copies — field mutation does not affect underlying data
        for elem in const_span:
            elem.translation = mochi.Real3(88.0, 88.0, 88.0)
        self.assertEqual(arr[0].translation, original_translation)
        self.assertEqual(arr[1].translation, mochi.Real3(4, 5, 6))

        # --- Mutable span of arithmetic (SpanReal) ---
        real_arr = mochi.DynamicArrayReal([10.0, 20.0, 30.0])
        real_span = mochi.SpanReal(real_arr)
        self.assertEqual(3, len(real_span))

        # __getitem__ returns a float (value copy — Python floats are immutable)
        x = real_span[0]
        self.assertEqual(type(x), float)
        self.assertEqual(x, 10.0)

        # __iter__ yields float values
        vals = list(real_span)
        self.assertEqual([10.0, 20.0, 30.0], vals)

        # __setitem__ works on mutable arithmetic spans
        real_span[0] = 99.0
        self.assertEqual(real_arr[0], 99.0)

    def test_span_copy_returns_dynamic_array(self):
        """Span is a non-owning view; copy/deepcopy promote to an owning DynamicArray
        (const dropped on the element type).
        """
        backing = mochi.DynamicArrayInt([10, 20, 30])
        span = mochi.SpanConstInt(backing)

        for c in (copy.copy(span), copy.deepcopy(span)):
            self.assertIsInstance(c, mochi.DynamicArrayInt)
            self.assertEqual(list(c), [10, 20, 30])
            # Mutating the source must not affect the snapshot.
            backing[0] = 999
            self.assertEqual(c[0], 10)
            backing[0] = 10  # restore for next iteration
