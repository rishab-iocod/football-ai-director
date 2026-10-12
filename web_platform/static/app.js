// app.js — Live Broadcast Studio Client Logic

class BroadcastStudioApp {
  constructor() {
    this.ws = null;
    this.reconnectTimer = null;
    this.currentZoom = 1.0;
    this.currentMode = "DIRECTOR_AUTO";
    this.activeCam = "cam2";
    this.panX = 0.0;
    this.panY = 0.0;
    this.showPip = true;

    // Audio synthesizer context
    this.audioCtx = null;

    this.initElements();
    this.initEventListeners();
    this.initPitchRadar();
    this.connectWebSocket();
  }

  initElements() {
    // Top Bar
    this.sourceSelect = document.getElementById("sourceSelect");
    this.scoreHome = document.getElementById("scoreHome");
    this.scoreAway = document.getElementById("scoreAway");
    this.matchClock = document.getElementById("matchClock");
    this.frameCounter = document.getElementById("frameCounter");
    this.engineFps = document.getElementById("engineFps");

    // Transport
    this.btnPlay = document.getElementById("btnPlay");
    this.btnPause = document.getElementById("btnPause");
    this.btnStop = document.getElementById("btnStop");
    this.speedBtns = document.querySelectorAll(".speed-btn");

    // Program Monitor
    this.onAirCamTitle = document.getElementById("onAirCamTitle");
    this.directorPolicyBadge = document.getElementById("directorPolicyBadge");
    this.ptzZoomBadge = document.getElementById("ptzZoomBadge");
    this.btnTogglePip = document.getElementById("btnTogglePip");
    this.pipOverlay = document.getElementById("pipOverlay");
    this.broadcastToast = document.getElementById("broadcastToast");
    this.toastBadge = document.getElementById("toastBadge");
    this.toastText = document.getElementById("toastText");
    this.programViewport = document.getElementById("programViewport");

    // Event buttons
    this.btnTriggerGoalHome = document.getElementById("btnTriggerGoalHome");
    this.btnTriggerGoalAway = document.getElementById("btnTriggerGoalAway");
    this.btnTriggerTacticalFoul = document.getElementById("btnTriggerTacticalFoul");
    this.btnTriggerHardTackle = document.getElementById("btnTriggerHardTackle");

    // Multiview tiles
    this.camTiles = document.querySelectorAll(".cam-tile");
    this.cutBtns = document.querySelectorAll(".btn-cut");
    this.scoreCam1 = document.getElementById("scoreCam1");
    this.scoreCam2 = document.getElementById("scoreCam2");
    this.scoreCam3 = document.getElementById("scoreCam3");

    // PTZ Controls
    this.ptzModeDisplay = document.getElementById("ptzModeDisplay");
    this.modePills = document.querySelectorAll(".mode-pill");
    this.zoomSlider = document.getElementById("zoomSlider");
    this.zoomValueText = document.getElementById("zoomValueText");
    this.presetBtns = document.querySelectorAll(".preset-btn");
    this.dpadBtns = document.querySelectorAll(".dpad-btn");

    // Radar Canvas
    this.canvas = document.getElementById("pitchCanvas");
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null;

    // Telemetry stats
    this.ballSpeedKmh = document.getElementById("ballSpeedKmh");
    this.ballSpeedPx = document.getElementById("ballSpeedPx");
    this.ballTrajectoryStatus = document.getElementById("ballTrajectoryStatus");
    this.ballSourceMethod = document.getElementById("ballSourceMethod");
    this.carrierId = document.getElementById("carrierId");
    this.carrierTeamText = document.getElementById("carrierTeamText");
    this.playerCountVal = document.getElementById("playerCountVal");
    this.possHomePct = document.getElementById("possHomePct");
    this.possAwayPct = document.getElementById("possAwayPct");
    this.possFillHome = document.getElementById("possFillHome");
    this.possFillAway = document.getElementById("possFillAway");

    // Event Log
    this.eventsList = document.getElementById("eventsList");
    this.eventCountBadge = document.getElementById("eventCountBadge");

    // Decision Reason Banner
    this.decisionText = document.getElementById("decisionText");
    this.scorePillCam1 = document.getElementById("scorePillCam1");
    this.scorePillCam2 = document.getElementById("scorePillCam2");
    this.scorePillCam3 = document.getElementById("scorePillCam3");

    // Guide & Settings Modal
    this.guideModal = document.getElementById("guideModal");
    this.btnOpenGuide = document.getElementById("btnOpenGuide");
    this.btnCloseGuide = document.getElementById("btnCloseGuide");
    this.guideTabs = document.querySelectorAll(".guide-tab");
    this.tabPanes = document.querySelectorAll(".tab-pane");

    // Director Settings inputs
    this.cfgMinShot = document.getElementById("cfgMinShot");
    this.valMinShot = document.getElementById("valMinShot");
    this.cfgHysteresis = document.getElementById("cfgHysteresis");
    this.valHysteresis = document.getElementById("valHysteresis");
    this.cfgCenterBonus = document.getElementById("cfgCenterBonus");
    this.valCenterBonus = document.getElementById("valCenterBonus");
    this.cfgFallbackCam = document.getElementById("cfgFallbackCam");
  }

  initEventListeners() {
    // Guide Modal
    if (this.btnOpenGuide) {
      this.btnOpenGuide.addEventListener("click", () => {
        this.guideModal.style.display = "flex";
      });
    }
    if (this.btnCloseGuide) {
      this.btnCloseGuide.addEventListener("click", () => {
        this.guideModal.style.display = "none";
      });
    }
    if (this.guideModal) {
      this.guideModal.addEventListener("click", (e) => {
        if (e.target === this.guideModal) this.guideModal.style.display = "none";
      });
    }

    // Modal Tabs
    this.guideTabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        this.guideTabs.forEach((t) => t.classList.remove("active"));
        this.tabPanes.forEach((p) => p.classList.remove("active"));
        tab.classList.add("active");
        const pane = document.getElementById(tab.dataset.tab);
        if (pane) pane.classList.add("active");
      });
    });

    // Settings Sliders
    if (this.cfgMinShot) {
      this.cfgMinShot.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        this.valMinShot.textContent = `${val.toFixed(1)}s`;
        this.sendPost("/api/director/config", { min_shot_seconds: val });
      });
    }
    if (this.cfgHysteresis) {
      this.cfgHysteresis.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        this.valHysteresis.textContent = `${val.toFixed(1)}s`;
        this.sendPost("/api/director/config", { hysteresis_seconds: val });
      });
    }
    if (this.cfgCenterBonus) {
      this.cfgCenterBonus.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        this.valCenterBonus.textContent = `+${val} pts`;
        this.sendPost("/api/director/config", { score_center_max: val });
      });
    }
    if (this.cfgFallbackCam) {
      this.cfgFallbackCam.addEventListener("change", (e) => {
        const val = parseInt(e.target.value);
        this.sendPost("/api/director/config", { fallback_cam: val });
      });
    }

    // Source dropdown change with transition lock
    this.sourceSelect.addEventListener("change", async (e) => {
      const srcKey = e.target.value;
      this.sourceSelect.disabled = true;
      try {
        await this.sendPost("/api/source/select", { source_key: srcKey });
        this.refreshAllStreams();
      } finally {
        setTimeout(() => {
          this.sourceSelect.disabled = false;
        }, 600);
      }
    });

    // Transport buttons
    this.btnPlay.addEventListener("click", async () => {
      await this.sendPost("/api/playback/start", {});
      this.btnPlay.classList.add("active");
      this.btnPause.classList.remove("active");
      this.refreshAllStreams();
    });

    this.btnPause.addEventListener("click", async () => {
      await this.sendPost("/api/playback/pause", {});
      this.btnPause.classList.add("active");
      this.btnPlay.classList.remove("active");
    });

    this.btnStop.addEventListener("click", async () => {
      await this.sendPost("/api/playback/stop", {});
      this.btnPlay.classList.remove("active");
      this.btnPause.classList.remove("active");
      setTimeout(async () => {
        await this.sendPost("/api/playback/start", {});
        this.btnPlay.classList.add("active");
        this.refreshAllStreams();
      }, 250);
    });

    this.speedBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        this.speedBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const spd = parseFloat(btn.dataset.speed);
        this.sendPost("/api/playback/speed", { speed: spd });
      });
    });

    // Toggle PiP
    this.btnTogglePip.addEventListener("click", () => {
      this.showPip = !this.showPip;
      this.pipOverlay.style.display = this.showPip ? "block" : "none";
    });

    // Event Triggers
    this.btnTriggerGoalHome.addEventListener("click", () => {
      this.sendPost("/api/trigger/goal", { team: 1 });
      this.playGoalChime();
    });

    this.btnTriggerGoalAway.addEventListener("click", () => {
      this.sendPost("/api/trigger/goal", { team: 2 });
      this.playGoalChime();
    });

    this.btnTriggerTacticalFoul.addEventListener("click", () => {
      this.sendPost("/api/trigger/foul", { severity: "TACTICAL_FOUL" });
      this.playWhistleChime();
    });

    this.btnTriggerHardTackle.addEventListener("click", () => {
      this.sendPost("/api/trigger/foul", { severity: "HARD_TACKLE" });
      this.playWhistleChime();
    });

    // Camera Switcher Tiles Click
    this.camTiles.forEach((tile) => {
      tile.addEventListener("click", () => {
        const cam = tile.dataset.cam;
        this.cutToCamera(cam);
      });
    });

    // Switcher Bus Buttons
    this.cutBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const cam = btn.dataset.cam;
        this.cutToCamera(cam);
      });
    });

    // Focus Mode Pills
    this.modePills.forEach((pill) => {
      pill.addEventListener("click", () => {
        this.modePills.forEach((p) => p.classList.remove("active"));
        pill.classList.add("active");
        this.currentMode = pill.dataset.mode;
        this.ptzModeDisplay.textContent = pill.textContent.toUpperCase();
        this.sendPtzUpdate({ mode: this.currentMode });
      });
    });

    // Zoom Slider
    this.zoomSlider.addEventListener("input", (e) => {
      const z = parseFloat(e.target.value);
      this.currentZoom = z;
      this.zoomValueText.textContent = `${z.toFixed(1)}x`;
      this.ptzZoomBadge.textContent = `ZOOM: ${z.toFixed(1)}x`;
      this.sendPtzUpdate({ zoom: z });
    });

    // Zoom Presets
    this.presetBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const z = parseFloat(btn.dataset.zoom);
        this.zoomSlider.value = z;
        this.currentZoom = z;
        this.zoomValueText.textContent = `${z.toFixed(1)}x`;
        this.ptzZoomBadge.textContent = `ZOOM: ${z.toFixed(1)}x`;
        this.sendPtzUpdate({ zoom: z });
      });
    });

    // Manual Pan D-Pad
    this.dpadBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        if (btn.classList.contains("up")) this.panY = Math.max(-1.0, this.panY - 0.2);
        else if (btn.classList.contains("down")) this.panY = Math.min(1.0, this.panY + 0.2);
        else if (btn.classList.contains("left")) this.panX = Math.max(-1.0, this.panX - 0.2);
        else if (btn.classList.contains("right")) this.panX = Math.min(1.0, this.panX + 0.2);
        else if (btn.classList.contains("center")) {
          this.panX = 0.0;
          this.panY = 0.0;
        }
        this.currentMode = "MANUAL_PTZ";
        this.updateModeUI("MANUAL_PTZ");
        this.sendPtzUpdate({ pan_x: this.panX, pan_y: this.panY, mode: "MANUAL_PTZ" });
      });
    });

    // Interactive Drag to Pan on Viewport
    let isDragging = false;
    let startX = 0, startY = 0;

    this.programViewport.addEventListener("mousedown", (e) => {
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
    });

    window.addEventListener("mouseup", () => { isDragging = false; });

    this.programViewport.addEventListener("mousemove", (e) => {
      if (!isDragging) return;
      const dx = (e.clientX - startX) / 300.0;
      const dy = (e.clientY - startY) / 300.0;
      startX = e.clientX;
      startY = e.clientY;

      this.panX = Math.max(-1.0, Math.min(1.0, this.panX - dx));
      this.panY = Math.max(-1.0, Math.min(1.0, this.panY - dy));

      this.currentMode = "MANUAL_PTZ";
      this.updateModeUI("MANUAL_PTZ");
      this.sendPtzUpdate({ pan_x: this.panX, pan_y: this.panY, mode: "MANUAL_PTZ" });
    });

    // Keyboard Shortcuts for Instant Camera Switching
    window.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      const key = e.key.toUpperCase();
      if (key === "1") this.cutToCamera("cam1");
      else if (key === "2") this.cutToCamera("cam2");
      else if (key === "3") this.cutToCamera("cam3");
      else if (key === "B") this.cutToCamera("ball_focus");
      else if (key === "P") this.cutToCamera("player_focus");
      else if (key === "0" || key === "W") this.cutToCamera("panoramic");
      else if (key === "A") this.cutToCamera("auto");
      else if (e.code === "Space") {
        e.preventDefault();
        this.btnPause.click();
      }
    });
  }

  refreshAllStreams() {
    const ts = Date.now();
    const progImg = document.getElementById("programFeedImg");
    if (progImg) progImg.src = `/stream/program?t=${ts}`;
    const pipImg = document.getElementById("roiPatchImg");
    if (pipImg) pipImg.src = `/stream/roi_patch?t=${ts}`;

    const tileImgs = document.querySelectorAll(".tile-stream-box img");
    tileImgs.forEach((img) => {
      const base = img.src.split("?")[0];
      img.src = `${base}?t=${ts}`;
    });
  }

  cutToCamera(camId) {
    this.sendPost("/api/director/cut", { camera_id: camId });
    this.updateSwitcherUI(camId);
  }

  updateSwitcherUI(camId) {
    // Update Switcher buttons
    this.cutBtns.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.cam === camId);
    });

    // Update active tile border
    this.camTiles.forEach((tile) => {
      const match = (camId === "auto" && tile.id === "tileCam2") || (tile.dataset.cam === camId);
      tile.classList.toggle("active-live", match);
    });

    if (camId === "auto") {
      this.directorPolicyBadge.textContent = "AUTO AI DIRECTOR";
      this.onAirCamTitle.textContent = "AI AUTO • WIDE TACTICAL";
    } else if (camId === "ball_focus") {
      this.directorPolicyBadge.textContent = "MANUAL: BALL LOCK";
      this.onAirCamTitle.textContent = "BALL FOCUS CAM";
    } else if (camId === "player_focus") {
      this.directorPolicyBadge.textContent = "MANUAL: HERO LOCK";
      this.onAirCamTitle.textContent = "HERO PLAYER CAM";
    } else if (camId === "panoramic") {
      this.directorPolicyBadge.textContent = "MANUAL: STADIUM PANORAMA";
      this.onAirCamTitle.textContent = "STADIUM PANORAMA ON-AIR";
    } else {
      const num = camId.replace("cam", "");
      this.directorPolicyBadge.textContent = `MANUAL CUT: CAM ${num}`;
      this.onAirCamTitle.textContent = `CAM ${num} ON-AIR`;
    }
  }

  updateModeUI(mode) {
    this.modePills.forEach((p) => {
      p.classList.toggle("active", p.dataset.mode === mode);
    });
    this.ptzModeDisplay.textContent = mode.replace("_", " ");
  }

  sendPtzUpdate(params) {
    this.sendPost("/api/director/ptz", params);
  }

  async sendPost(url, body) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      return await res.json();
    } catch (e) {
      console.warn("API Error:", e);
    }
  }

  // ---------------------------------------------------------------- WebSocket
  connectWebSocket() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/ws`;

    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      console.log("[Studio WS] Connected");
    };

    this.ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === "telemetry") {
          this.handleTelemetry(msg);
        } else if (msg.type === "event") {
          this.handleEvent(msg);
        } else if (msg.type === "init") {
          if (msg.recent_events) {
            msg.recent_events.forEach((ev) => this.prependEventCard(ev));
          }
        }
      } catch (e) {
        console.error("WS Parse error:", e);
      }
    };

    this.ws.onclose = () => {
      console.warn("[Studio WS] Closed. Reconnecting in 2s...");
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = setTimeout(() => this.connectWebSocket(), 2000);
    };

    this.ws.onerror = () => {
      this.ws.close();
    };
  }

  handleTelemetry(data) {
    // Scoreboard & Timing
    if (data.score) {
      this.scoreHome.textContent = data.score.team_1;
      this.scoreAway.textContent = data.score.team_2;
    }
    this.matchClock.textContent = data.match_time || "00:00";
    this.frameCounter.textContent = `F#${data.frame_index || 0}`;
    this.engineFps.textContent = (data.fps || 25.0).toFixed(1);

    // Live Director Decision Reason & Scores
    if (data.director_reason && this.decisionText) {
      this.decisionText.textContent = data.director_reason;
    }

    // Program Bus and Multiview Tiles Active Highlighting
    const activeView = data.active_program_view || "auto";
    const activeNum = data.active_cam_id || 2;

    this.cutBtns.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.cam === activeView);
    });

    this.camTiles.forEach((tile) => {
      let isLive = false;
      if (activeView === "auto") {
        isLive = (tile.dataset.cam === `cam${activeNum}`);
      } else {
        isLive = (tile.dataset.cam === activeView);
      }
      tile.classList.toggle("active-live", isLive);
    });

    if (this.onAirCamTitle) {
      if (activeView === "ball_focus") {
        this.onAirCamTitle.textContent = "BALL FOCUS CAM • AUTO-PTZ";
        if (this.directorPolicyBadge) this.directorPolicyBadge.textContent = "MANUAL: BALL LOCK";
      } else if (activeView === "player_focus") {
        this.onAirCamTitle.textContent = "HERO PLAYER CAM • HERO LOCK";
        if (this.directorPolicyBadge) this.directorPolicyBadge.textContent = "MANUAL: HERO LOCK";
      } else if (activeView === "auto") {
        const labels = { 1: "LEFT WING", 2: "WIDE TACTICAL", 3: "RIGHT WING / GOAL" };
        this.onAirCamTitle.textContent = `CAM ${activeNum} • ${labels[activeNum] || "ON-AIR"}`;
        if (this.directorPolicyBadge) this.directorPolicyBadge.textContent = "AUTO AI DIRECTOR";
      } else {
        const num = activeView.replace("cam", "");
        this.onAirCamTitle.textContent = `CAM ${num} MANUAL LOCK`;
        if (this.directorPolicyBadge) this.directorPolicyBadge.textContent = `MANUAL CUT: CAM ${num}`;
      }
    }

    // Camera Scores in Switcher
    if (data.camera_scores) {
      data.camera_scores.forEach((cs) => {
        if (cs.cam_id === "cam1") {
          if (this.scoreCam1) this.scoreCam1.textContent = `SC: ${cs.score}`;
          if (this.scorePillCam1) {
            this.scorePillCam1.textContent = `Cam 1: ${cs.score}`;
            this.scorePillCam1.classList.toggle("active", cs.is_live);
          }
        } else if (cs.cam_id === "cam2") {
          if (this.scoreCam2) this.scoreCam2.textContent = `SC: ${cs.score}`;
          if (this.scorePillCam2) {
            this.scorePillCam2.textContent = `Cam 2: ${cs.score}`;
            this.scorePillCam2.classList.toggle("active", cs.is_live);
          }
        } else if (cs.cam_id === "cam3") {
          if (this.scoreCam3) this.scoreCam3.textContent = `SC: ${cs.score}`;
          if (this.scorePillCam3) {
            this.scorePillCam3.textContent = `Cam 3: ${cs.score}`;
            this.scorePillCam3.classList.toggle("active", cs.is_live);
          }
        }
      });
    }

    // Ball Kinematics
    if (data.ball) {
      this.ballSpeedKmh.textContent = (data.ball.speed_kmh || 0).toFixed(1);
      this.ballSpeedPx.textContent = `${(data.ball.speed_px_f || 0).toFixed(1)} px/f`;
      
      const isTracked = data.ball.tracked;
      this.ballTrajectoryStatus.textContent = isTracked ? "TRACKED" : "SEARCHING";
      this.ballTrajectoryStatus.className = `stat-status ${isTracked ? "green" : "amber"}`;
      this.ballSourceMethod.textContent = data.ball.is_extrapolated ? "Physics Flight Bridge" : `Direct (${data.ball.source})`;
    }

    // Possession & Carrier
    if (data.possession) {
      const p = data.possession;
      const t1 = p.team_1_pct || 50;
      const t2 = p.team_2_pct || 50;
      this.possHomePct.textContent = `${t1}%`;
      this.possAwayPct.textContent = `${t2}%`;
      this.possFillHome.style.width = `${t1}%`;
      this.possFillAway.style.width = `${t2}%`;

      this.carrierId.textContent = p.carrier_id ? `#${p.carrier_id}` : "#--";
      this.carrierTeamText.textContent = p.team ? `Team ${p.team}` : "Open Play";
    }

    this.playerCountVal.textContent = data.players_count || 0;

    // 2D Pitch Radar
    if (data.radar) {
      this.drawPitchRadar(data.radar);
    }
  }

  handleEvent(msg) {
    const evType = msg.event_type;
    const data = msg.data;

    if (evType === "goal") {
      this.showToast("⚽ GOAL!", data.description, "gold");
      this.playGoalChime();
      this.prependEventCard({
        type: "GOAL",
        description: data.description,
        timestamp: data.timestamp,
        badge_class: "goal",
      });
    } else if (evType === "foul") {
      const icon = data.card === "YELLOW_CARD" ? "🟨" : "⚠️";
      this.showToast(`${icon} FOUL DETECTED`, data.description, "red");
      this.playWhistleChime();
      this.prependEventCard({
        type: "FOUL",
        description: data.description,
        timestamp: data.timestamp,
        badge_class: "foul",
      });
    } else if (evType === "cut") {
      this.prependEventCard({
        type: "CAMERA CUT",
        description: `Cut to Cam ${data.to_cam}: ${data.reason}`,
        timestamp: Date.now() / 1000,
        badge_class: "cut",
      });
    }
  }

  showToast(badge, text, color) {
    this.toastBadge.textContent = badge;
    this.toastText.textContent = text;
    this.broadcastToast.style.borderLeftColor = color === "gold" ? "var(--accent-gold)" : "var(--accent-red)";
    this.toastBadge.style.color = color === "gold" ? "var(--accent-gold)" : "var(--accent-red)";
    this.broadcastToast.classList.add("visible");

    setTimeout(() => {
      this.broadcastToast.classList.remove("visible");
    }, 4000);
  }

  prependEventCard(ev) {
    const emptyMsg = this.eventsList.querySelector(".empty-events-msg");
    if (emptyMsg) emptyMsg.remove();

    const d = new Date((ev.timestamp || Date.now() / 1000) * 1000);
    const timeStr = d.toTimeString().split(" ")[0];

    const card = document.createElement("div");
    card.className = `event-card-item ${ev.badge_class || "goal"}`;
    card.innerHTML = `
      <div class="event-card-top">
        <span class="event-type-badge">${ev.type || "EVENT"}</span>
        <span class="event-time">${timeStr}</span>
      </div>
      <div class="event-desc">${ev.description || ""}</div>
    `;

    this.eventsList.insertBefore(card, this.eventsList.firstChild);

    // Update count
    const total = this.eventsList.querySelectorAll(".event-card-item").length;
    this.eventCountBadge.textContent = `${total} EVENTS`;
  }

  // ---------------------------------------------------------------- 2D Radar Canvas
  initPitchRadar() {
    if (!this.canvas || !this.ctx) return;
    this.drawPitchRadar({ ball: null, players: [] });
  }

  drawPitchRadar(radarData) {
    if (!this.ctx || !this.canvas) return;
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    // Pitch Background
    ctx.fillStyle = "#0c281e";
    ctx.fillRect(0, 0, w, h);

    // Outer boundary & Pitch markings
    ctx.strokeStyle = "rgba(74, 222, 128, 0.4)";
    ctx.lineWidth = 1.5;

    const pad = 12;
    const pw = w - pad * 2;
    const ph = h - pad * 2;

    ctx.strokeRect(pad, pad, pw, ph);

    // Halfway line
    ctx.beginPath();
    ctx.moveTo(w / 2, pad);
    ctx.lineTo(w / 2, h - pad);
    ctx.stroke();

    // Center circle
    ctx.beginPath();
    ctx.arc(w / 2, h / 2, 28, 0, Math.PI * 2);
    ctx.stroke();

    // Penalty Boxes
    // Left Box
    ctx.strokeRect(pad, h / 2 - 40, 48, 80);
    // Right Box
    ctx.strokeRect(w - pad - 48, h / 2 - 40, 48, 80);

    // Draw Players
    const players = radarData.players || [];
    players.forEach((p) => {
      const px = pad + p.x * pw;
      const py = pad + p.y * ph;

      ctx.beginPath();
      ctx.arc(px, py, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = p.team === 1 ? "#06b6d4" : "#ef4444";
      ctx.fill();

      if (p.has_ball) {
        ctx.strokeStyle = "#fbbf24";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });

    // Draw Ball
    if (radarData.ball) {
      const bx = pad + radarData.ball[0] * pw;
      const by = pad + radarData.ball[1] * ph;

      // Glowing ball
      ctx.beginPath();
      ctx.arc(bx, by, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#f59e0b";
      ctx.fill();

      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  // ---------------------------------------------------------------- Audio Synthesis
  getAudioContext() {
    if (!this.audioCtx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new AudioCtx();
    }
    if (this.audioCtx.state === "suspended") {
      this.audioCtx.resume();
    }
    return this.audioCtx;
  }

  playGoalChime() {
    try {
      const ctx = this.getAudioContext();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "triangle";
      osc.frequency.setValueAtTime(440, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.3);
      gain.gain.setValueAtTime(0.15, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.6);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.6);
    } catch (e) {}
  }

  playWhistleChime() {
    try {
      const ctx = this.getAudioContext();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(1200, ctx.currentTime);
      osc.frequency.setValueAtTime(1400, ctx.currentTime + 0.1);
      gain.gain.setValueAtTime(0.12, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.35);
    } catch (e) {}
  }
}

// Instantiate on DOM load
window.addEventListener("DOMContentLoaded", () => {
  window.studioApp = new BroadcastStudioApp();
});
