export default {
  key: "fourcast-livive-net",
  serve: { dir: "." },
  viewport: { width: 1280, height: 800 },
  durationMs: 30000,
  settleMs: 1500,

  async ready(page) {
    await page.waitForFunction(
      () => {
        const status = document.querySelector("#status-pill")?.textContent || "";
        return !/Initializing/i.test(status) && document.querySelectorAll("#gallery-container img").length > 0;
      },
      null,
      { timeout: 90000 },
    ).catch(() => {});
    await page.waitForTimeout(1500);
  },

  beats: [
    {
      label: "Four paradigms on one holdout cat",
      action: { click: "#random-sample-btn" },
      wait: 3000,
      shot: "four-paradigms",
      thumb: true,
    },
    {
      label: "HOG gradients and the pixel pipeline",
      action: { scrollToSelector: "#card-hog" },
      wait: 2000,
      shot: "hog-pipeline",
    },
    {
      label: "Filter the holdout gallery",
      action: { click: 'button[data-filter="dog"]' },
      wait: 1800,
      shot: "holdout-gallery",
    },
    {
      label: "Benchmark table and holdout accuracy",
      action: [
        { scrollTo: 0 },
        { click: 'a[data-tab="analytics"]' },
      ],
      wait: 2500,
      shot: "benchmark-metrics",
    },
    {
      label: "The four paradigms explained",
      action: { click: 'a[data-tab="background"]' },
      wait: 2200,
      shot: "paradigm-theory",
    },
    {
      label: "Back to the playground",
      action: [
        { click: 'a[data-tab="playground"]' },
        { scrollTo: 0 },
      ],
      wait: 1500,
    },
  ],

  thumbFrom: "four-paradigms",
};
