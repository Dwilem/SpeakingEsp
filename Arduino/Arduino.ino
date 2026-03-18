#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_camera.h"
#include "driver/i2s.h"

// ── Config ────────────────────────────────────────────────────────────────────
const char* SSID        = "VR_House2";
const char* PASS        = "YOUR_PASS";
const char* SERVER_IP   = "192.168.0.101";   // your PC's IP
const int   AUDIO_PORT  = 5005;
const int   VIDEO_PORT  = 5006;

// ── I2S mic pins (INMP441 / SPH0645) ─────────────────────────────────────────
#define I2S_WS   15
#define I2S_SD   13
#define I2S_SCK  14

// ── Camera pin map (ESP32-S3 CAM Dev Board N16R8 — OV3660/OV2640) ──────────
#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD     4  // SDA
#define CAM_PIN_SIOC     5  // SCL
#define CAM_PIN_D7      16  // Y9
#define CAM_PIN_D6      17  // Y8
#define CAM_PIN_D5      18  // Y7
#define CAM_PIN_D4      12  // Y6
#define CAM_PIN_D3      10  // Y5
#define CAM_PIN_D2       8  // Y4
#define CAM_PIN_D1       9  // Y3
#define CAM_PIN_D0      11  // Y2
#define CAM_PIN_VSYNC    6
#define CAM_PIN_HREF     7
#define CAM_PIN_PCLK    13

WiFiUDP audioUdp;
WiFiUDP videoUdp;

// ─────────────────────────────────────────────────────────────────────────────
void initI2S() {
  i2s_config_t cfg = {
    .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate          = 16000,
    .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,  // mic outputs 24-bit in 32-bit frame
    .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count        = 4,
    .dma_buf_len          = 512,
    .use_apll             = false,
    .tx_desc_auto_clear   = false,
    .fixed_mclk           = 0
  };
  i2s_pin_config_t pins = {
    .bck_io_num   = I2S_SCK,
    .ws_io_num    = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = I2S_SD
  };
  i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pins);
}

void initCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = CAM_PIN_D0;
  config.pin_d1       = CAM_PIN_D1;
  config.pin_d2       = CAM_PIN_D2;
  config.pin_d3       = CAM_PIN_D3;
  config.pin_d4       = CAM_PIN_D4;
  config.pin_d5       = CAM_PIN_D5;
  config.pin_d6       = CAM_PIN_D6;
  config.pin_d7       = CAM_PIN_D7;
  config.pin_xclk     = CAM_PIN_XCLK;
  config.pin_pclk     = CAM_PIN_PCLK;
  config.pin_vsync    = CAM_PIN_VSYNC;
  config.pin_href     = CAM_PIN_HREF;
  config.pin_sscb_sda = CAM_PIN_SIOD;
  config.pin_sscb_scl = CAM_PIN_SIOC;
  config.pin_pwdn     = CAM_PIN_PWDN;
  config.pin_reset    = CAM_PIN_RESET;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_QVGA;   // 320×240 — safe over UDP
  config.jpeg_quality = 12;               // 0=best, 63=worst
  config.fb_count     = 2;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
  esp_camera_init(&config);
}

// ─────────────────────────────────────────────────────────────────────────────
// Audio task — runs on core 1
// Reads 512 int32 samples, strips upper 8 bits (mic format), sends as int16
// ─────────────────────────────────────────────────────────────────────────────
void audioTask(void*) {
  const int SAMPLES = 480;   // 16000 Hz × 30 ms
  int32_t  raw[SAMPLES];
  int16_t  pcm[SAMPLES];

  while (true) {
    size_t bytesRead = 0;
    i2s_read(I2S_NUM_0, raw, sizeof(raw), &bytesRead, portMAX_DELAY);

    int count = bytesRead / sizeof(int32_t);
    for (int i = 0; i < count; i++)
      pcm[i] = (int16_t)(raw[i] >> 14);   // 24-bit value sitting in bits [31:8]

    audioUdp.beginPacket(SERVER_IP, AUDIO_PORT);
    audioUdp.write((uint8_t*)pcm, count * sizeof(int16_t));
    audioUdp.endPacket();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Video task — runs on core 0
// JPEG frames can be larger than the UDP MTU (1472 bytes), so we chunk them.
// Each chunk is prefixed with a 6-byte header: [frame_id u16][chunk_idx u16][total_chunks u16]
// The server reassembles before saving.
// ─────────────────────────────────────────────────────────────────────────────
void videoTask(void*) {
  const int CHUNK = 1400;   // leave headroom inside the 1472-byte Ethernet payload
  uint16_t  frameId = 0;

  while (true) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) { vTaskDelay(10); continue; }

    uint16_t totalChunks = (fb->len + CHUNK - 1) / CHUNK;
    for (uint16_t i = 0; i < totalChunks; i++) {
      uint8_t  buf[CHUNK + 6];
      uint16_t offset = i * CHUNK;
      uint16_t len    = min((size_t)CHUNK, fb->len - offset);

      // Header
      memcpy(buf + 0, &frameId,     2);
      memcpy(buf + 2, &i,           2);
      memcpy(buf + 4, &totalChunks, 2);
      memcpy(buf + 6, fb->buf + offset, len);

      videoUdp.beginPacket(SERVER_IP, VIDEO_PORT);
      videoUdp.write(buf, len + 6);
      videoUdp.endPacket();
    }

    esp_camera_fb_return(fb);
    frameId++;
    vTaskDelay(1000);   // ~30 fps cap
  }
}

// ─────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  WiFi.begin(SSID, PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\nWiFi connected: " + WiFi.localIP().toString());

  initI2S();
  initCamera();

  audioUdp.begin(AUDIO_PORT);
  videoUdp.begin(VIDEO_PORT);

  // Audio on core 1, video on core 0 — keeps them from starving each other
  xTaskCreatePinnedToCore(audioTask, "audio", 4096, NULL, 1, NULL, 1);
  xTaskCreatePinnedToCore(videoTask, "video", 8192, NULL, 1, NULL, 0);
}

void loop() { vTaskDelay(1000); }