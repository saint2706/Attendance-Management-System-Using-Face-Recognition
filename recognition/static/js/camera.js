"use strict";

/**
 * CameraManager centralises MediaStream lifecycle management for templates that
 * provide camera previews or capture workflows.
 *
 * Basic usage:
 * ```javascript
 * import { CameraManager } from "{% static 'js/camera.js' %}";
 * const manager = new CameraManager();
 * await manager.start(document.querySelector("video"));
 * // ... later when the UI closes
 * manager.stop();
 * ```
 *
 * The manager ensures the same stream instance is reused across multiple
 * invocations and guarantees that tracks are stopped during cleanup.
 */
export class CameraManager {
  constructor(defaultConstraints) {
    this._defaultConstraints =
      defaultConstraints || CameraManager.defaultConstraints;
    this._stream = null;
    this._initialising = null;
    this._videoElements = new Set();
  }

  /**
   * Retrieve an active stream and optionally attach it to a video element.
   *
   * @param {HTMLVideoElement} [videoElement] - Element that should render the stream.
   * @param {MediaStreamConstraints} [constraints] - Custom constraints for `getUserMedia`.
   * @returns {Promise<MediaStream>} The active MediaStream instance.
   */
  async start(videoElement, constraints) {
    const stream = await this.getStream(constraints);
    if (videoElement) {
      this.attach(videoElement);
    }
    return stream;
  }

  /**
   * Return a MediaStream, initialising it if necessary.
   *
   * @param {MediaStreamConstraints} [constraints] - Optional constraints override.
   * @returns {Promise<MediaStream>} The ready MediaStream instance.
   */
  async getStream(constraints) {
    const desiredConstraints = constraints || this._defaultConstraints;
    if (constraints) {
      this._defaultConstraints = constraints;
    }

    if (this._stream) {
      return this._stream;
    }

    if (!this._initialising) {
      this._initialising = this._requestStream(desiredConstraints);
    }

    return this._initialising;
  }

  /**
   * Attach the active stream to a video element.
   *
   * @param {HTMLVideoElement} videoElement - Element that should render the stream.
   */
  attach(videoElement) {
    if (!videoElement) {
      return;
    }

    this._videoElements.add(videoElement);

    if (!this._stream) {
      return;
    }

    // Ensure the element always reflects the active stream.
    videoElement.srcObject = this._stream;
    if (typeof videoElement.play === "function") {
      const playResult = videoElement.play();
      if (playResult && typeof playResult.catch === "function") {
        playResult.catch(() => {
          /* Ignore autoplay errors caused by browser policies. */
        });
      }
    }
  }

  /**
   * Detach a video element from the active stream.
   *
   * @param {HTMLVideoElement} videoElement - Element previously passed to `attach`.
   */
  detach(videoElement) {
    if (!videoElement) {
      return;
    }

    this._videoElements.delete(videoElement);
    if (typeof videoElement.pause === "function") {
      videoElement.pause();
    }
    videoElement.srcObject = null;
  }

  /**
   * Stop all tracks and release the managed MediaStream.
   */
  stop() {
    this._videoElements.forEach((videoElement) => {
      if (typeof videoElement.pause === "function") {
        videoElement.pause();
      }
      videoElement.srcObject = null;
    });
    this._videoElements.clear();

    if (this._stream) {
      this._stream.getTracks().forEach((track) => track.stop());
      this._stream = null;
    }
    this._initialising = null;
  }

  /**
   * Dispose the manager and clean up all resources.
   */
  dispose() {
    this.stop();
  }

  async _requestStream(constraints) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      this._initialising = null;
      throw new Error("Camera access is not supported in this browser.");
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      this._stream = stream;
      return stream;
    } catch (error) {
      this._stream = null;
      throw error;
    } finally {
      this._initialising = null;
    }
  }

  /**
   * Default video constraints used when none are provided.
   */
  static get defaultConstraints() {
    return {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user",
      },
      audio: false,
    };
  }
}
