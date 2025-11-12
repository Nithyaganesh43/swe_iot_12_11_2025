#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

const char* ssid = "Sorry";
const char* password = "aaaaaaaa";

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

httpd_handle_t stream_httpd = NULL;

static esp_err_t esp_handler(httpd_req_t *req){
  const char* resp_str = "iamespcam";
  httpd_resp_send(req, resp_str, strlen(resp_str));
  return ESP_OK;
}

static esp_err_t stream_handler(httpd_req_t *req){
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  size_t jpg_len;
  uint8_t * jpg_buf;
  char part_buf[64];

  static const char* STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=frame";
  static const char* STREAM_BOUNDARY = "\r\n--frame\r\n";
  static const char* STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

  res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if(res != ESP_OK) return res;

  while(true){
    fb = esp_camera_fb_get();
    if(!fb){
      res = ESP_FAIL;
      break;
    }
    jpg_len = fb->len;
    jpg_buf = fb->buf;

    res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if(res == ESP_OK){
      int len = snprintf(part_buf, 64, STREAM_PART, jpg_len);
      res = httpd_resp_send_chunk(req, part_buf, len);
    }
    if(res == ESP_OK){
      res = httpd_resp_send_chunk(req, (const char *)jpg_buf, jpg_len);
    }
    esp_camera_fb_return(fb);
    if(res != ESP_OK) break;
  }
  return res;
}

void startCameraServer(){
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;

  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };
  httpd_uri_t esp_uri = {
    .uri       = "/esp",
    .method    = HTTP_GET,
    .handler   = esp_handler,
    .user_ctx  = NULL
  };

  if(httpd_start(&stream_httpd, &config) == ESP_OK){
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    httpd_register_uri_handler(stream_httpd, &esp_uri);
  }
}

void setup(){
  Serial.begin(115200);
  Serial.println("Starting...");

  WiFi.begin(ssid, password);
  while(WiFi.status()!=WL_CONNECTED){
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 10;
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;

  if(esp_camera_init(&config)!=ESP_OK){
    Serial.println("Camera init failed");
    return;
  }
  Serial.println("Camera ready");
  startCameraServer();
  Serial.println("Streaming at:");
  Serial.print("http://");
  Serial.print(WiFi.localIP());
  Serial.println("/stream");
}

void loop(){
  delay(1000);
}