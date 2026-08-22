/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace CADRobotExporter.UI
{
    public partial class ProgressIndicatorForm : Form
    {
        public ProgressIndicatorForm()
        {
            InitializeComponent();

            buttonCancel.Click += ButtonCancel_Click;

            Reset();
        }

        private void ButtonCancel_Click(object sender, EventArgs e)
        {
            labelCancelNotification.Text = "Attemping to cancel, please wait...";
            labelStep.ForeColor = System.Drawing.Color.Gray;
            labelSubSteps.ForeColor = System.Drawing.Color.Gray;
            progressBar.ForeColor = System.Drawing.Color.Gray;
            isCancelled = true;
        }

        public void Reset()
        {
            isCancelled = false;
            steps = new List<Step>();
            subSteps = new List<Step>();
            SetProgress(0, 10);
            labelSubSteps.Text = "";
            labelStep.Text = "";
            labelTimeElapsed.Text = "";
        }

        private class Step
        {
            public string Name;
            public string Text;
        }

        private volatile bool isCancelled;
        public bool IsCancelled => isCancelled;

        private List<Step> steps;
        private List<Step> subSteps;

        private string currentStepName;
        private string currentSubStepName;
        private int currentStepIndex;
        private int currentSubStepIndex;

        private DateTime startTime;

        public void StartTimer()
        {
            startTime = System.DateTime.Now;
            timer1.Interval = 500;
            timer1.Tick += new EventHandler(OnTick);
            timer1.Start();
        }

        private void OnTick(object sender, EventArgs e)
        {
            TimeSpan elapsed = System.DateTime.Now - startTime;
            int minutes = (int)elapsed.TotalMinutes;
            int seconds = elapsed.Seconds;
            if (minutes > 0)
            {
                labelTimeElapsed.Text = $"{minutes} minutes {seconds} seconds";
            }
            else
            {
                labelTimeElapsed.Text = $"{seconds} seconds";
            }
        }

        public void AddStep(string name, string text)
        {
            steps.Add(
                new Step
                {
                    Name = name,
                    Text = text,
                });
        }

        public void AddSubStep(string name, string text)
        {
            subSteps.Add(
                new Step
                {
                    Name = name,
                    Text = text,
                });
        }

        public void SetCurrentStep(string name)
        {
            Step step = steps.Find(st => st.Name == name);
            if (step != null)
            {
                currentStepIndex = steps.IndexOf(step);
                currentStepName = name;
                currentSubStepIndex = 0;
                UpdateProgress();
            }
        }

        public void SetCurrentSubstep(string name)
        {
            Step step = subSteps.Find(st => st.Name == name);
            if (step != null)
            {
                currentSubStepIndex = subSteps.IndexOf(step);
                currentSubStepName = name;
                UpdateProgress();
            }
        }

        private void SetProgress(int value, int max)
        {
            progressBar.Maximum = max;
            progressBar.Value = value;
        }

        private void UpdateProgress()
        {
            int maxValue = steps.Count * Math.Max(1, subSteps.Count);
            int currentValue = currentStepIndex * subSteps.Count + currentSubStepIndex;

            labelStep.Text = $"{steps[currentStepIndex].Text} ({currentStepIndex + 1} of {steps.Count})";

            string subStepText = "";
            for (int i = 0; i < subSteps.Count; i++)
            {
                bool isDone = currentSubStepIndex > i;

                string line = "";
                line += isDone ? "☑ " : "☐ ";
                line += subSteps[i].Text;
                line += currentSubStepIndex == i ? " ←" : "";
                subStepText += line + "\n";
            }

            labelSubSteps.Text = subStepText;
            SetProgress(currentValue, maxValue);
        }
    }
}
