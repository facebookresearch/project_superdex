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

# _____________________________________________________________________________


def _get_ndarray_primitive(p):
    if p.dtype in ["int", "int32", "int64"]:
        return "int"
    if p.dtype in ["uint", "uint32", "uint64"]:
        return "unsigned int"
    if p.dtype in ["float", "float32", "float64"]:
        return "real"
    raise AssertionError("Can't convert this primitive type.")


def _get_ndarray_typename(p):
    primitive = _get_ndarray_primitive(p)

    if primitive == "real":
        if len(p.shape) == 1 and len(p) < 4:
            return f"real{len(p)}"
        if len(p.shape) == 2 and p.shape[0] < 4 and p.shape[1] < 4:
            return f"Matrix{p.shape[0]}x{p.shape[1]}"

    if primitive == "real":
        if len(p.shape) == 1 and len(p) < 4:
            return f"int{len(p)}"

    shape = ", ".join(["%i" % d for d in p.shape])
    return f"NdArray<{primitive}, {shape}>"


def _format_ndarray_entry(x, primitive):
    if primitive == "int":
        return f"{x}" % x
    if primitive == "unsigned int":
        return f"0x{x:08x}u"
    if primitive == "real":
        return f"{x}_r"
    return ""


def export_ndarray(p, name=False):
    primitive = _get_ndarray_primitive(p)
    typename = _get_ndarray_typename(p)

    if len(p.shape) == 1:
        entries = [_format_ndarray_entry(x, primitive) % x for x in p]
        return typename + "{ " + ", ".join(entries) + " }"
    else:
        delim = ",\n" if name else ", "
        vals = delim.join([export_ndarray(x) for x in p])
        prefix = typename + (" %s = " % name) if name else ""
        posfix = ";" if name else ""
        return prefix + typename + ("{ %s }" % vals) + posfix


# _____________________________________________________________________________
