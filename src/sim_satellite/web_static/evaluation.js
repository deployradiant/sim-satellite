(function () {
  "use strict";

  function byId(id) {
    return document.getElementById(id);
  }

  function randomId(prefix) {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return `${prefix}-${window.crypto.randomUUID()}`;
    }
    return `${prefix}-${Date.now()}-${Math.floor(Math.random() * 1000000)}`;
  }

  function randomSeed() {
    if (window.crypto && typeof window.crypto.getRandomValues === "function") {
      const values = new Uint32Array(1);
      window.crypto.getRandomValues(values);
      return values[0];
    }
    return Math.floor(Math.random() * 4294967295);
  }

  function evaluationSessionId() {
    const key = "simSatelliteEvaluationSessionId";
    try {
      const existing = window.localStorage.getItem(key);
      if (existing) return existing;
      const created = randomId("session");
      window.localStorage.setItem(key, created);
      return created;
    } catch {
      return randomId("session");
    }
  }

  function createEvaluationController(deps) {
    const taskPreferenceButton = byId("eval-task-preference");
    const taskRealFakeButton = byId("eval-task-real-fake");
    const confidenceInput = byId("eval-confidence");
    const startButton = byId("eval-start");
    const count = byId("eval-count");
    const title = byId("eval-title");
    const images = byId("eval-images");
    const frameLeft = byId("eval-frame-left");
    const frameRight = byId("eval-frame-right");
    const preferenceActions = byId("eval-preference-actions");
    const realFakeActions = byId("eval-real-fake-actions");
    const leftButton = byId("eval-left");
    const tieButton = byId("eval-tie");
    const rightButton = byId("eval-right");
    const badPairButton = byId("eval-bad-pair");
    const realButton = byId("eval-real");
    const fakeButton = byId("eval-fake");
    const skipButton = byId("eval-skip");

    let task = "preference";
    let activeTrial = null;
    let submittingVote = false;

    function init() {
      taskPreferenceButton.addEventListener("click", () => setTask("preference"));
      taskRealFakeButton.addEventListener("click", () => setTask("real_fake"));
      startButton.addEventListener("click", startTrial);
      leftButton.addEventListener("click", () => submitChoice("left"));
      tieButton.addEventListener("click", () => submitChoice("tie"));
      rightButton.addEventListener("click", () => submitChoice("right"));
      badPairButton.addEventListener("click", () => submitChoice("bad_pair"));
      realButton.addEventListener("click", () => submitChoice("real"));
      fakeButton.addEventListener("click", () => submitChoice("fake"));
      skipButton.addEventListener("click", () => submitChoice("skip"));
      setTask("preference");
    }

    function setTask(nextTask) {
      task = nextTask;
      const isPreference = task === "preference";
      taskPreferenceButton.setAttribute("aria-pressed", String(isPreference));
      taskRealFakeButton.setAttribute("aria-pressed", String(!isPreference));
      preferenceActions.hidden = !isPreference;
      realFakeActions.hidden = isPreference;
      title.textContent = isPreference ? "Preference trial" : "Real/Fake trial";
      activeTrial = null;
      setAnswerDisabled(true);
      renderPlaceholder("No trial yet");
    }

    function renderCount(state) {
      const evaluation = state && state.evaluation ? state.evaluation : {};
      const votes = Number(evaluation.votes) || 0;
      const label = votes === 1 ? "1 vote recorded" : `${votes} votes recorded`;
      count.textContent = evaluation.logPath ? `${label} | ${evaluation.logPath}` : label;
    }

    function setAnswerDisabled(disabled) {
      [
        leftButton,
        tieButton,
        rightButton,
        badPairButton,
        realButton,
        fakeButton,
        skipButton,
      ].forEach((button) => {
        button.disabled = disabled;
      });
    }

    function renderPlaceholder(text) {
      images.classList.remove("single");
      frameRight.closest(".evaluation-frame").hidden = false;
      setFrame(frameLeft, "A", null, text);
      setFrame(frameRight, "B", null, text);
    }

    function setFrame(frame, label, item, placeholder) {
      const parent = frame.closest(".evaluation-frame");
      parent.querySelector(".evaluation-frame-label").textContent = label;
      if (!item) {
        frame.replaceChildren(deps.emptyMessage(placeholder));
        return;
      }
      const image = document.createElement("img");
      image.alt = label;
      image.src = item.src;
      frame.replaceChildren(image);
    }

    function renderPreferenceTrial(left, right) {
      images.classList.remove("single");
      frameRight.closest(".evaluation-frame").hidden = false;
      setFrame(frameLeft, "A", left, "");
      setFrame(frameRight, "B", right, "");
    }

    function renderRealFakeTrial(shown) {
      images.classList.add("single");
      frameRight.closest(".evaluation-frame").hidden = true;
      setFrame(frameLeft, "Image", shown, "");
    }

    function shuffledPair(first, second) {
      return Math.random() < 0.5 ? [first, second] : [second, first];
    }

    function realImage(sampleIndex) {
      return {
        src: deps.sampleImageSrc(sampleIndex),
        record: {
          kind: "real",
          label: "Ground truth",
        },
      };
    }

    function generatedImage(payload, checkpoint, seed) {
      return {
        src: payload.image,
        record: {
          kind: "generated",
          checkpointId: payload.checkpointId || checkpoint.id,
          checkpointLabel: payload.checkpointLabel || checkpoint.label,
          seed,
        },
      };
    }

    async function startTrial() {
      await runAction("Starting evaluation trial...", async () => {
        const state = await deps.refreshStateMetadata();
        const targets = deps.availableCheckpoints(state);
        if (!targets.length) throw new Error("No checkpoint is ready");
        const sample = deps.currentSample(state);
        const seed = randomSeed();
        const steps = deps.requestedSteps(state);
        if (task === "preference") {
          await startPreferenceTrial(targets, sample, seed, steps);
        } else {
          await startRealFakeTrial(targets, sample, seed, steps);
        }
      });
    }

    async function startPreferenceTrial(targets, sample, seed, steps) {
      const body = deps.generationRequestBody(steps, seed);
      const generationTargets = targets.slice(0, Math.min(targets.length, 2));
      const generatedItems = [];
      const totalSteps = steps * generationTargets.length;
      deps.showProgress(totalSteps);

      let completedSteps = 0;
      for (const checkpoint of generationTargets) {
        const payload = await deps.requestGenerationStream(body, checkpoint, {
          checkpoint,
          offset: completedSteps,
          totalSteps,
          seed,
        });
        if (!payload) throw new Error(`${checkpoint.label} ended before the image was ready`);
        generatedItems.push(generatedImage(payload, checkpoint, seed));
        completedSteps += payload.steps || steps;
        deps.updateProgress(completedSteps, totalSteps);
      }

      const pair =
        generatedItems.length > 1
          ? shuffledPair(generatedItems[0], generatedItems[1])
          : shuffledPair(generatedItems[0], realImage(sample.index));

      renderPreferenceTrial(pair[0], pair[1]);
      activeTrial = {
        task: "preference",
        trialId: randomId("trial"),
        sessionId: evaluationSessionId(),
        sampleIndex: sample.index,
        sampleName: sample.name,
        seed,
        left: pair[0].record,
        right: pair[1].record,
      };
      setAnswerDisabled(false);
      deps.setStatus("Preference trial ready");
    }

    async function startRealFakeTrial(targets, sample, seed, steps) {
      const showFake = Math.random() < 0.5;
      let shown = realImage(sample.index);

      if (showFake) {
        const checkpoint = targets[0];
        deps.showProgress(steps);
        const payload = await deps.requestGenerationStream(
          deps.generationRequestBody(steps, seed),
          checkpoint,
          {
            checkpoint,
            offset: 0,
            totalSteps: steps,
            seed,
          },
        );
        if (!payload) throw new Error(`${checkpoint.label} ended before the image was ready`);
        shown = generatedImage(payload, checkpoint, seed);
      } else {
        deps.hideProgress();
      }

      renderRealFakeTrial(shown);
      activeTrial = {
        task: "real_fake",
        trialId: randomId("trial"),
        sessionId: evaluationSessionId(),
        sampleIndex: sample.index,
        sampleName: sample.name,
        seed,
        shown: shown.record,
      };
      setAnswerDisabled(false);
      deps.setStatus("Real/Fake trial ready");
    }

    async function runAction(startMessage, action) {
      activeTrial = null;
      setAnswerDisabled(true);
      startButton.disabled = true;
      deps.setGenerationControlsDisabled(true);
      deps.setStatus(startMessage);
      try {
        await action();
      } catch (error) {
        deps.failProgress();
        deps.setStatus(error.message, true);
      } finally {
        startButton.disabled = false;
        deps.setGenerationControlsDisabled(false);
      }
    }

    async function submitChoice(choice) {
      if (!activeTrial || submittingVote) {
        deps.setStatus("Start a trial first", true);
        return;
      }
      const trial = activeTrial;
      activeTrial = null;
      submittingVote = true;
      setAnswerDisabled(true);

      const payload = {
        ...trial,
        choice,
        confidence: Number(confidenceInput.value),
      };
      const response = await fetch("/api/evaluation/votes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const result = await response.json().catch(() => ({}));
      submittingVote = false;
      if (!response.ok) {
        activeTrial = trial;
        setAnswerDisabled(false);
        deps.setStatus(result.detail || result.error || "Could not record vote", true);
        return;
      }

      deps.onVoteRecorded(result.votes);
      deps.setStatus("Vote recorded");
    }

    return {
      init,
      renderCount,
    };
  }

  window.createEvaluationController = createEvaluationController;
})();
