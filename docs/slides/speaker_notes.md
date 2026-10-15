# Speaker Notes

What to say on each slide. Aim for about a minute per slide, so the whole talk runs 15 to 18 minutes before questions.

**1. Title.** Introduce yourselves and the project. Point at the list on the right: these are real commits from the newest part of the history, scored by our model. By the end you'll see how it decides which ones are risky.

**2. Review time is limited.** Reviewers can't look at everything with the same care. Ask the audience which of the two commits they'd review first. Most people pick the big one, and that intuition is roughly what we're trying to teach a model, except with reasons attached.

**3. What the tool does.** Walk through the four steps on the left, then read out the first reason in the terminal output. Mention that this commit really was labelled buggy later on.

**4. How it works.** Follow the arrows: mine the repository, compute features and labels, split by time, compare models, explain, and serve it in an app. Every box is a command in the project.

**5. The data.** 2,788 commits, about 22% buggy. Point at the chart and say that most of the activity was in the early years and the bug rate moves around, which is the reason for a time-based split later.

**6. Features.** Five groups of features. Read out the example commit: 91 lines added across 3 files, and it was the author's first commit. Stress that history and experience only use earlier commits.

**7. SZZ.** Explain the three steps simply: find fixes, blame the lines they changed, label whoever wrote those lines. Then be upfront about the weaknesses. Knowing them makes the rest of the results easier to trust.

**8. What separates buggy commits.** The median buggy commit adds 12 lines, the median clean one adds 3. But all the boxes overlap, so no single feature decides it. That's why we need a model.

**9. Clustering.** Without seeing any labels, K-Means found a group of cross-cutting commits that are buggy about 41% of the time, roughly double the others. DBSCAN's outliers told the same story.

**10. Testing it the honest way.** Explain why a random split cheats. Point at the folds: each one trains on the past and validates on what comes next. The test set was used once, at the very end.

**11. 13 algorithms.** XGBoost won on cross-validation, but say clearly that the top seven are basically tied. The interesting finding is that the data limits us more than the algorithm does.

**12. Imbalance and overfitting.** Left side: without rebalancing, the model misses about 70% of buggy commits at the default threshold. Right side: a deep decision tree memorises the training data. Ensembles and tuning fix that.

**13. Results.** Read the three numbers, then spend time on the 48% card, because it's the most practical result. Mention the honest catch: at this threshold it flags 44% of commits, which a real team would tune.

**14. Why is this commit risky?** Explain SHAP in one sentence: it splits the prediction into a sum of contributions. Read the sentence in the red box. LIME agreed most of the time.

**15. What drives risk overall.** Lines added dominates. Every other direction also makes sense, which is a good sign the model learned something real.

**16. Accuracy isn't the whole story.** This is our favourite finding. Two models with almost the same score, but one of them gave backwards explanations because of correlated features. If you only looked at accuracy you'd never notice.

**17. The app.** Switch to the live demo here (see the demo plan in the Q&A notes). If the demo can't run, talk through the screenshot.

**18. Limitations and next steps.** Be honest about noisy labels and having one project. The learning curve shows more of the same data won't help, so the next steps are better labels and features.

**19. Thank you.** Give the GitHub link and invite questions.
