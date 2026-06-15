#include "vmsys.h"
#include "vmgraph.h"
#include "vmchset.h"
#include "vmstdlib.h"
#include "vmcamera.h"
#include <net.h>
//#include <cstdint>

//From https://github.com/freshtechyy/NCNN-Deployment-image-classification-example
#include "image_classifier_opt.id.h"
#include "image_classifier_opt.mem.h"

VMINT layer_hdl[1];
VMUINT8* layer_buf = 0;
VMINT screen_w = 0;
VMINT screen_h = 0;
VM_CAMERA_HANDLE handle_c;
VMWCHAR w_frame[64];
VMWCHAR w_nn[64];
VMWCHAR percent_lut[101][5];
VMWCHAR number_lut[1000][5];
static int g_frame_len;
static int g_nn_len;
static int g_frame_value_x;
static int g_nn_value_x;
static const VMWCHAR percent_symbol[] = { '%', 0 };

static ncnn::Net g_classifier;
static int g_classifier_loaded = 0;

#define NN_W 32
#define NN_H 32

static ncnn::Mat in;

static unsigned char src_x[NN_W];
static unsigned char src_y[NN_H];

static const char* class_names[10] = {
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

static int g_pred_class = -1;
static float g_pred_score = 0.f;

static int g_top_class[3] = {-1, -1, -1};
static float g_top_score[3] = {0.f, 0.f, 0.f};

static int g_load_param_ret = -999;
static int g_load_model_ret = -999;
static int g_input_ret = -999;
static int g_extract_ret = -999;
static int g_output_w = 0;
static int g_output_h = 0;
static int g_output_c = 0;
static int g_output_total = 0;
static int g_best_logit1000 = 0;

void handle_sysevt(VMINT message, VMINT param);
void draw_text(int l, int x, int y, char* str, unsigned short c);
static void draw_percent_fast(int layer, int x, int y, float score, unsigned short color565);
void init_resize_lut();
void init_number_lut();

static inline unsigned char claim(int x)
{
    return (unsigned char)(x > 255 ? 255 : (x < 0 ? 0 : x));
}

void classify_image(ncnn::Mat& input)
{
    if (!g_classifier_loaded)
    {
        g_pred_class = -1;
        g_pred_score = 0.f;
        return;
    }

    g_pred_class = -1;
    g_pred_score = 0.f;

    g_input_ret = -999;
    g_extract_ret = -999;
    g_output_w = 0;
    g_output_h = 0;
    g_output_c = 0;
    g_output_total = 0;
    g_best_logit1000 = 0;

    ncnn::Extractor ex = g_classifier.create_extractor();
    ex.set_light_mode(true);

    g_input_ret = ex.input(image_classifier_opt_param_id::BLOB_input, input);
    if (g_input_ret != 0)
        return;

    ncnn::Mat output;

    g_extract_ret = ex.extract(image_classifier_opt_param_id::BLOB_output, output);
    if (g_extract_ret != 0)
        return;

    g_output_w = output.w;
    g_output_h = output.h;
    g_output_c = output.c;
    g_output_total = output.w * output.h * output.c;

    ncnn::Mat flat = output.reshape(g_output_total);

    if (flat.w <= 0)
        return;

int n = flat.w;
if (n > 10)
    n = 10;

/* Find max logit for stable softmax */
float max_logit = flat[0];

for (int i = 1; i < n; i++)
{
    if (flat[i] > max_logit)
        max_logit = flat[i];
}

/* Calculate softmax sum */
float sum = 0.f;

for (int i = 0; i < n; i++)
{
    sum += expf(flat[i] - max_logit);
}

/* Reset top 3 */
for (int i = 0; i < 3; i++)
{
    g_top_class[i] = -1;
    g_top_score[i] = 0.f;
}

/* Find top 3 classes */
for (int i = 0; i < n; i++)
{
    float prob = 0.f;

    if (sum > 0.f)
        prob = expf(flat[i] - max_logit) / sum;

    for (int k = 0; k < 3; k++)
    {
        if (g_top_class[k] < 0 || prob > g_top_score[k])
        {
            for (int m = 2; m > k; m--)
            {
                g_top_class[m] = g_top_class[m - 1];
                g_top_score[m] = g_top_score[m - 1];
            }

            g_top_class[k] = i;
            g_top_score[k] = prob;
            break;
        }
    }
}

/* Keep old single prediction variables too */
g_pred_class = g_top_class[0];
g_pred_score = g_top_score[0];

if (g_pred_class >= 0 && g_pred_class < 10)
    g_best_logit1000 = (int)(flat[g_pred_class] * 1000.f);
else
    g_best_logit1000 = 0;

    }

void cam_message_callback(const vm_cam_notify_data_t* notify_data, void* user_data)
{

    if (!notify_data)
        return;

    if (notify_data->cam_message != VM_CAM_PREVIEW_FRAME_RECEIVED)
        return;

    vm_cam_frame_data_t frame;

    if (vm_camera_get_frame(handle_c, &frame) != VM_CAM_SUCCESS)
        return;

    unsigned int time1 = vm_get_tick_count();

    unsigned char* s = (unsigned char*)frame.pixtel_data;
    unsigned short* layer_buf_s = (unsigned short*)layer_buf;

    int pix = 0;

    for (unsigned int y = 0; y < 240; y++)
    {
        for (unsigned int x = 0; x < 240; x += 2)
        {
            int u0 = *s++ - 128;
            int y0 = *s++ - 16;
            int v  = *s++ - 128;
            int y2 = *s++ - 16;

            int tmp = 298 * y0 + 128;
            int vv  = 409 * v;
            int uv  = -100 * u0 - 208 * v;
            int uu  = 516 * u0;

            unsigned char r = claim((tmp + vv) >> 8);
            unsigned char g = claim((tmp + uv) >> 8);
            unsigned char b = claim((tmp + uu) >> 8);

            layer_buf_s[pix] = VM_COLOR_888_TO_565(r, g, b);

            tmp = 298 * y2 + 128;

            r = claim((tmp + vv) >> 8);
            g = claim((tmp + uv) >> 8);
            b = claim((tmp + uu) >> 8);

            layer_buf_s[pix + 1] = VM_COLOR_888_TO_565(r, g, b);

            pix += 2;
        }
    }

    /* =========================
       Build 32x32 NN image
       ========================= */

    float* pc0 = in.channel(0);
    float* pc1 = in.channel(1);
    float* pc2 = in.channel(2);

    unsigned short* fb = (unsigned short*)layer_buf;

for (int ny = 0; ny < NN_H; ny++)
{
    int sy = src_y[ny];

    for (int nx = 0; nx < NN_W; nx++)
    {
        int sx = src_x[nx];

        unsigned short p = fb[sy * 240 + sx];

        unsigned char r = (unsigned char)((((p >> 11) & 0x1F) * 255) / 31);
        unsigned char g = (unsigned char)((((p >> 5) & 0x3F) * 255) / 63);
        unsigned char b = (unsigned char)(((p & 0x1F) * 255) / 31);

        int idx = ny * NN_W + nx;

        pc0[idx] = (b - 127.5f) * (1.0f / 127.5f);
        pc1[idx] = (g - 127.5f) * (1.0f / 127.5f);
        pc2[idx] = (r - 127.5f) * (1.0f / 127.5f);

    }
}


    unsigned int time2 = vm_get_tick_count();

    static int infer_counter = 0;
    infer_counter++;

    if ((infer_counter % 10) == 0)
    {
        classify_image(in);
    }

    unsigned int time3 = vm_get_tick_count();

    vm_graphic_fill_rect(layer_buf, 0, 240, screen_w, screen_h - 240, VM_COLOR_WHITE, VM_COLOR_WHITE);

char line[80];

if (g_top_class[0] >= 0)
{
    char line1[64];
    char line2[64];
    char line3[64];

    VMWCHAR w_line1[64];
    VMWCHAR w_line2[64];
    VMWCHAR w_line3[64];

    vm_graphic_color color = { VM_COLOR_888_TO_565(0, 0, 0) };
    vm_graphic_setcolor(&color);

    sprintf(line1, "1. %s %d%%", class_names[g_top_class[0]], (int)(g_top_score[0] * 100.0f + 0.5f));
    sprintf(line2, "2. %s %d%%", class_names[g_top_class[1]], (int)(g_top_score[1] * 100.0f + 0.5f));
    sprintf(line3, "3. %s %d%%", class_names[g_top_class[2]], (int)(g_top_score[2] * 100.0f + 0.5f));

    vm_ascii_to_ucs2(w_line1, sizeof(w_line1), line1);
    vm_ascii_to_ucs2(w_line2, sizeof(w_line2), line2);
    vm_ascii_to_ucs2(w_line3, sizeof(w_line3), line3);

    vm_graphic_textout_to_layer(layer_hdl[0], 0, 262, w_line1, vm_wstrlen(w_line1));
    vm_graphic_textout_to_layer(layer_hdl[0], 0, 282, w_line2, vm_wstrlen(w_line2));
    vm_graphic_textout_to_layer(layer_hdl[0], 0, 302, w_line3, vm_wstrlen(w_line3));
}
else
{
    vm_graphic_fill_rect(layer_buf, 0, 240, screen_w, screen_h - 240, VM_COLOR_WHITE, VM_COLOR_WHITE);
    sprintf(line, "No result");
    draw_text(layer_hdl[0], 2, 262, line, VM_COLOR_888_TO_565(0, 0, 0));

}

    vm_graphic_color color = {VM_COLOR_888_TO_565(0, 0, 0)};
    vm_graphic_setcolor(&color);

    int frame_ms = (int)(time2 - time1);
    if (frame_ms > 999) frame_ms = 999;

    vm_graphic_textout_to_layer(layer_hdl[0], 0, 242, w_frame, g_frame_len);
    vm_graphic_textout_to_layer(layer_hdl[0], g_frame_value_x, 242, number_lut[frame_ms], vm_wstrlen(number_lut[frame_ms]));

    int nn_ms = (int)(time3 - time2);
    if (nn_ms > 999) nn_ms = 999;

    vm_graphic_textout_to_layer(layer_hdl[0], 100, 242, w_nn, g_nn_len);
    vm_graphic_textout_to_layer(layer_hdl[0], 100 + g_nn_value_x, 242, number_lut[nn_ms], vm_wstrlen(number_lut[nn_ms]));

    vm_graphic_flush_layer(layer_hdl, 1);
}

void vm_main(void)
{
    screen_w = vm_graphic_get_screen_width();
    screen_h = vm_graphic_get_screen_height();

    layer_hdl[0] = vm_graphic_create_layer(0,0,screen_w,screen_h,-1);
    layer_buf = vm_graphic_get_layer_buffer(layer_hdl[0]);

    vm_reg_sysevt_callback(handle_sysevt);
    vm_switch_power_saving_mode(turn_off_mode);

    in.create(NN_W, NN_H, 3);

    init_number_lut();
    init_resize_lut();

    vm_ascii_to_ucs2(w_frame, 64, "Frame: ");
    vm_ascii_to_ucs2(w_nn, 64, "Ncnn: ");

    g_frame_value_x = vm_graphic_get_string_width(w_frame);
    g_nn_value_x = vm_graphic_get_string_width(w_nn);
    g_frame_len = vm_wstrlen(w_frame);
    g_nn_len    = vm_wstrlen(w_nn);

    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 1;

    g_classifier.opt = opt;

g_load_param_ret = g_classifier.load_param(image_classifier_opt_param_bin);
g_load_model_ret = g_classifier.load_model(image_classifier_opt_bin);

if (g_load_param_ret >= 0 && g_load_model_ret >= 0)
    g_classifier_loaded = 1;
else
    g_classifier_loaded = 0;


    vm_create_camera_instance((VM_CAMERA_ID)vm_camera_get_main_camera_id(), &handle_c);

    vm_camera_register_notify(handle_c, cam_message_callback, 0);

    vm_cam_size_t sz = {240,240};
    vm_camera_set_preview_size(handle_c, &sz);

    vm_camera_preview_start(handle_c);

}

void draw_text(int l, int x, int y, char* str, unsigned short c)
{
    VMWCHAR wstr[100] = {0};

    vm_ascii_to_ucs2(wstr, 200, str);

    vm_graphic_color color = { c };
    vm_graphic_setcolor(&color);

    vm_graphic_textout_to_layer(l, x, y, wstr, vm_wstrlen(wstr));
}

void handle_sysevt(VMINT message, VMINT param)
{
    switch (message)
    {
    case VM_MSG_CREATE:
    case VM_MSG_ACTIVE:
        break;

    case VM_MSG_PAINT:
        break;

    case VM_MSG_INACTIVE:
        break;

    case VM_MSG_QUIT:
        g_classifier.clear();
        vm_release_camera_instance(handle_c);
        break;
    }
}

void init_resize_lut()
{
    for (int i = 0; i < NN_W; i++)
        src_x[i] = (unsigned char)((i * 240) / NN_W);

    for (int i = 0; i < NN_H; i++)
        src_y[i] = (unsigned char)((i * 240) / NN_H);
}

void init_number_lut()
{
    for (int n = 0; n < 1000; n++)
    {
        int pos = 0;

        if (n >= 100)
        {
            number_lut[n][pos++] = '0' + (n / 100);
            number_lut[n][pos++] = '0' + ((n / 10) % 10);
            number_lut[n][pos++] = '0' + (n % 10);
        }
        else if (n >= 10)
        {
            number_lut[n][pos++] = '0' + (n / 10);
            number_lut[n][pos++] = '0' + (n % 10);
        }
        else
        {
            number_lut[n][pos++] = '0' + n;
        }

        number_lut[n][pos] = 0;
    }
}

