/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;
using System.Threading;
using System.Windows.Forms;

namespace CADRobotExporter.UI
{
    public class ProgressIndicatorManager : IDisposable
    {
        private ProgressIndicatorForm form;
        private Thread uiThread;
        private readonly ManualResetEvent formReady = new ManualResetEvent(false);
        private bool isDisposed;

        public void Show()
        {
            if (uiThread != null && uiThread.IsAlive)
            {
                return;
            }

            formReady.Reset();

            uiThread = new Thread(() =>
            {
                form = new ProgressIndicatorForm();
                form.HandleCreated += (s, e) => formReady.Set();
                form.FormClosed += (s, e) => Application.ExitThread();
                Application.Run(form);
            });

            uiThread.SetApartmentState(ApartmentState.STA);
            uiThread.IsBackground = true;
            uiThread.Start();

            // Wait for form to be ready
            formReady.WaitOne();
            StartTimer();
        }

        public void Close()
        {
            InvokeOnForm(() => form.Close());
        }

        public void Reset()
        {
            InvokeOnForm(() => form.Reset());
        }

        public void AddStep(string name, string text)
        {
            InvokeOnForm(() => form.AddStep(name, text));
        }

        public void AddSteps(IEnumerable<(string name, string text)> steps)
        {
            InvokeOnForm(() =>
            {
                foreach (var (name, text) in steps)
                {
                    form.AddStep(name, text);
                }
            });
        }

        public void AddSubStep(string name, string text)
        {
            InvokeOnForm(() => form.AddSubStep(name, text));
        }

        public void AddSubSteps(IEnumerable<(string name, string text)> subSteps)
        {
            InvokeOnForm(() =>
            {
                foreach (var (name, text) in subSteps)
                {
                    form.AddSubStep(name, text);
                }
            });
        }

        public void SetCurrentStep(string name)
        {
            InvokeOnForm(() => form.SetCurrentStep(name));
        }

        public void SetCurrentSubStep(string name)
        {
            InvokeOnForm(() => form.SetCurrentSubstep(name));
        }

        public void StartTimer()
        {
            InvokeOnForm(() => form.StartTimer());
        }

        private void InvokeOnForm(Action action)
        {
            if (form == null || form.IsDisposed || !form.IsHandleCreated)
            {
                return;
            }

            if (form.InvokeRequired)
            {
                form.Invoke(action);
            }
            else
            {
                action();
            }
        }

        public bool IsCancelled
        {
            get
            {
                if (form == null || form.IsDisposed)
                    return true;  // Treat as cancelled if form is gone

                return form.IsCancelled;
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!isDisposed)
            {
                return;
            }

            if (form != null && !form.IsDisposed && form.IsHandleCreated)
            {
                form.Invoke(new Action(() => form.Close()));
            }

            formReady.Dispose();

            isDisposed = true;
        }
    }
}
