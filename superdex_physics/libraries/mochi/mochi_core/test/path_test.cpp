/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/path.h>

using namespace mochi;

TEST(Path, EnsureTrailingSlash) {
  EXPECT_STREQ("some/path/", path::EnsureTrailingSlash("some/path").c_str());
  EXPECT_STREQ("some/path/", path::EnsureTrailingSlash("some/path/").c_str());
  EXPECT_STREQ("some\\path\\", path::EnsureTrailingSlash("some\\path\\").c_str());
}

TEST(Path, IsAbsolute) {
  // Unix style absolute
  EXPECT_TRUE(path::IsAbsolutePath("~/some/path"));
  EXPECT_TRUE(path::IsAbsolutePath("/some/path"));

  // Windows style absolute
  EXPECT_TRUE(path::IsAbsolutePath("C:\\some\\path"));
  EXPECT_TRUE(path::IsAbsolutePath("\\\\network\\path"));

  // Relative
  EXPECT_FALSE(path::IsAbsolutePath("file"));
  EXPECT_FALSE(path::IsAbsolutePath("some/path"));
  EXPECT_FALSE(path::IsAbsolutePath("./file"));
  EXPECT_FALSE(path::IsAbsolutePath("./some/path"));
  EXPECT_FALSE(path::IsAbsolutePath("../file"));
  EXPECT_FALSE(path::IsAbsolutePath("../some/path"));
}

TEST(Path, GetFullPath) {
  // Test with absolute paths (should return as-is)
  EXPECT_EQ("/absolute/path", path::GetFullPath("/absolute/path", "/root/dir"));
  EXPECT_EQ("C:/windows/path", path::GetFullPath("C:/windows/path", "/root/dir"));
  EXPECT_EQ("\\\\network\\share", path::GetFullPath("\\\\network\\share", "/root/dir"));

  // Test with relative paths (should join with root)
  EXPECT_EQ("/root/dir/relative/path", path::GetFullPath("relative/path", "/root/dir"));
  EXPECT_EQ("/root/dir/file.txt", path::GetFullPath("file.txt", "/root/dir"));

  // Test with root paths with and without trailing slash
  EXPECT_EQ("/root/dir/file.txt", path::GetFullPath("file.txt", "/root/dir"));
  EXPECT_EQ("/root/dir/file.txt", path::GetFullPath("file.txt", "/root/dir/"));
}

TEST(Path, GetRelativePath) {
  // Test with relative paths (should return as-is)
  EXPECT_EQ("relative/path", path::GetRelativePath("relative/path", "/root/dir"));
  EXPECT_EQ("file.txt", path::GetRelativePath("file.txt", "/root/dir"));

  // Test with absolute paths that have a clear relative relationship
  // Case 1: Child directory relative to parent
  EXPECT_EQ("child", path::GetRelativePath("/parent/child", "/parent"));
  EXPECT_EQ("child/file.txt", path::GetRelativePath("/parent/child/file.txt", "/parent"));

  // Case 2: Sibling directories
  EXPECT_EQ("../sibling2", path::GetRelativePath("/parent/sibling2", "/parent/sibling1"));
  EXPECT_EQ(
      "../sibling2/file.txt",
      path::GetRelativePath("/parent/sibling2/file.txt", "/parent/sibling1"));

  // Case 3: Windows-style paths (in generic form)
  EXPECT_EQ("child", path::GetRelativePath("C:/parent/child", "C:/parent"));
  EXPECT_EQ("../sibling2", path::GetRelativePath("C:/parent/sibling2", "C:/parent/sibling1"));
}

TEST(Path, ScanDirectoryForFiles) {
  // Create empty directory.
  auto root = CreateTempDirectory("scan_directory_test", test::ExpectOK{});

  // Expect no files.
  auto files = path::ScanDirectoryForFiles(root.Path(), "", true, test::ExpectOK{});
  EXPECT_EQ(0, files.size());

  // Add some files
  WriteFile(root.Path() / "file1.aaa", "", test::ExpectOK{});
  WriteFile(root.Path() / "file2.bbb", "", test::ExpectOK{});
  WriteFile(root.Path() / "subdir/file3.aaa", "", test::ExpectOK{});
  WriteFile(root.Path() / "subdir/nested/file4.mochi.bbb", "", test::ExpectOK{}); // compound ext
  WriteFile(root.Path() / "subdir/nested/file5.mochi.aaa", "", test::ExpectOK{}); // compound ext
  WriteFile(root.Path() / "subdir/nested/file6.ccc", "", test::ExpectOK{});

  // Scan the root directory (recursive)
  files = path::ScanDirectoryForFiles(root.Path(), "", true, test::ExpectOK{});
  EXPECT_EQ(6, files.size());
  EXPECT_STREQ("file1.aaa", files[0].string().c_str());
  EXPECT_STREQ("file2.bbb", files[1].string().c_str());
  EXPECT_STREQ("subdir/file3.aaa", files[2].string().c_str());
  EXPECT_STREQ("subdir/nested/file4.mochi.bbb", files[3].string().c_str());
  EXPECT_STREQ("subdir/nested/file5.mochi.aaa", files[4].string().c_str());
  EXPECT_STREQ("subdir/nested/file6.ccc", files[5].string().c_str());

  // Scan the root directory (non-recursive)
  files = path::ScanDirectoryForFiles(root.Path(), "", false, test::ExpectOK{});
  EXPECT_EQ(2, files.size());
  EXPECT_STREQ("file1.aaa", files[0].string().c_str());
  EXPECT_STREQ("file2.bbb", files[1].string().c_str());

  // Scan a subfolder (recursive)
  files = path::ScanDirectoryForFiles(root.Path() / "subdir", "", true, test::ExpectOK{});
  EXPECT_EQ(4, files.size());
  EXPECT_STREQ("file3.aaa", files[0].string().c_str());
  EXPECT_STREQ("nested/file4.mochi.bbb", files[1].string().c_str());
  EXPECT_STREQ("nested/file5.mochi.aaa", files[2].string().c_str());
  EXPECT_STREQ("nested/file6.ccc", files[3].string().c_str());

  // Scan a subfolder (non-recursive)
  files = path::ScanDirectoryForFiles(root.Path() / "subdir", "", false, test::ExpectOK{});
  EXPECT_EQ(1, files.size());
  EXPECT_STREQ("file3.aaa", files[0].string().c_str());

  // Scan root for *.aaa files (recursive)
  files = path::ScanDirectoryForFiles(root.Path(), ".aaa", true, test::ExpectOK{});
  EXPECT_EQ(3, files.size());
  EXPECT_STREQ("file1.aaa", files[0].string().c_str());
  EXPECT_STREQ("subdir/file3.aaa", files[1].string().c_str());
  EXPECT_STREQ("subdir/nested/file5.mochi.aaa", files[2].string().c_str());

  // Scan root for *.bbb files (recursive)
  files = path::ScanDirectoryForFiles(root.Path(), ".bbb", true, test::ExpectOK{});
  EXPECT_EQ(2, files.size());
  EXPECT_STREQ("file2.bbb", files[0].string().c_str());
  EXPECT_STREQ("subdir/nested/file4.mochi.bbb", files[1].string().c_str());

  // Scan subdir for *.aaa and *.bbb files (recursive)
  files = path::ScanDirectoryForFiles(root.Path() / "subdir", ".aaa|.bbb", true, test::ExpectOK{});
  EXPECT_EQ(3, files.size());
  EXPECT_STREQ("file3.aaa", files[0].string().c_str());
  EXPECT_STREQ("nested/file4.mochi.bbb", files[1].string().c_str());
  EXPECT_STREQ("nested/file5.mochi.aaa", files[2].string().c_str());

  // Scan root for compound extension (recursive)
  files = path::ScanDirectoryForFiles(root.Path(), ".mochi.aaa", true, test::ExpectOK{});
  EXPECT_EQ(1, files.size());
  EXPECT_STREQ("subdir/nested/file5.mochi.aaa", files[0].string().c_str());
}
