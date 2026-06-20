#include "vmsys.h"
#include "vmgraph.h"
#include "vmchset.h"
#include "vmstdlib.h"
#include "vmcamera.h"
#include <net.h>

// [CHANGED] Use MNIST int8 headers instead of image_classifier_opt
#include "mnist-int8.id.h"
#include "mnist-int8.mem.h"

VMINT layer_hdl[1];
VMUINT8* layer_buf = 0;
VMINT screen_w = 0;
VMINT screen_h = 0;
VM_CAMERA_HANDLE handle_c;
VMWCHAR w_frame[64];
VMWCHAR w_nn[64];
VMWCHAR w_no_result[64];
VMWCHAR number_lut[1000][5];
static int g_frame_len;
static int g_nn_len;
static int g_no_result_len;
static int g_frame_value_x;
static int g_nn_value_x;
static ncnn::Net g_classifier;
static int g_classifier_loaded = 0;

// [CHANGED] MNIST input is 28x28
#define NN_W 28
#define NN_H 28

#define MIN_CONFIDENCE 0.50f
#define MIN_MARGIN     0.20f

static ncnn::Mat in;
static unsigned char src_x[NN_W];
static unsigned char src_y[NN_H];

// [CHANGED] MNIST digit class names: 0–9
static const VMWCHAR class_names_w[10][4] = {
    { '0', 0 },
    { '1', 0 },
    { '2', 0 },
    { '3', 0 },
    { '4', 0 },
    { '5', 0 },
    { '6', 0 },
    { '7', 0 },
    { '8', 0 },
    { '9', 0 }
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
void init_resize_lut();
void init_number_lut();
void draw_top_class_line_ucs2(int rank, int top_index, int y);

static inline unsigned char claim(int x)
{
    return (unsigned char)(x > 255 ? 255 : (x < 0 ? 0 : x));
}

void classify_image(ncnn::Mat& input)
{
    g_pred_class = -1;
    g_pred_score = 0.f;

    for (int i = 0; i < 3; i++)
    {
        g_top_class[i] = -1;
        g_top_score[i] = 0.f;
    }

    if (!g_classifier_loaded)
    {
        g_pred_class = -1;
        g_pred_score = 0.f;
        return;
    }

    g_input_ret = -999;
    g_extract_ret = -999;
    g_output_w = 0;
    g_output_h = 0;
    g_output_c = 0;
    g_output_total = 0;
    g_best_logit1000 = 0;

    ncnn::Extractor ex = g_classifier.create_extractor();
    ex.set_light_mode(true);

    // [CHANGED] Use mnist_int8_param_id blob names: BLOB_data (input), BLOB_fc (output)
    g_input_ret = ex.input(mnist_int8_param_id::BLOB_data, input);
    if (g_input_ret != 0)
        return;

    ncnn::Mat output;

    g_extract_ret = ex.extract(mnist_int8_param_id::BLOB_fc, output);
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
       Build 28x28 grayscale NN image for MNIST
       Normalization: pixel / 255.0  (mean=0, scale=1/255, matches ESP32 reference)
       NOTE: MNIST was trained on WHITE digit on BLACK background.
             If your camera shows dark digits on white paper, invert with:
             pc0[idx] = 1.0f - gray / 255.0f;
       ========================= */

    // [CHANGED] Single grayscale channel only (was 3-channel RGB)
    float* pc0 = in.channel(0);

    unsigned short* fb = (unsigned short*)layer_buf;

    for (int ny = 0; ny < NN_H; ny++)
    {
        int sy = src_y[ny];

        for (int nx = 0; nx < NN_W; nx++)
        {
            int sx = src_x[nx];

            unsigned short p = fb[sy * 240 + sx];

            unsigned char r = (unsigned char)((((p >> 11) & 0x1F) * 255) / 31);
            unsigned char g = (unsigned char)((((p >> 5)  & 0x3F) * 255) / 63);
            unsigned char b = (unsigned char)(((p & 0x1F) * 255) / 31);

            // [CHANGED] Convert RGB to grayscale (BT.601 luma coefficients)
            float gray = 0.299f * r + 0.587f * g + 0.114f * b;

            int idx = ny * NN_W + nx;

            // [CHANGED] MNIST normalization: [0,255] -> [0.0, 1.0]
            // For dark digit on bright background, use: pc0[idx] = 1.0f - gray / 255.0f;

            //pc0[idx] = gray / 255.0f;
            pc0[idx] = 1.0f - gray / 255.0f;   // invert for dark digit on white background
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

    vm_graphic_color color = { VM_COLOR_888_TO_565(0, 0, 0) };
    vm_graphic_setcolor(&color);

    if (g_top_class[0] >= 0 &&
        g_top_score[0] >= MIN_CONFIDENCE &&
        (g_top_score[0] - g_top_score[1]) >= MIN_MARGIN)
    {
        draw_top_class_line_ucs2(1, 0, 262);
        draw_top_class_line_ucs2(2, 1, 282);
        draw_top_class_line_ucs2(3, 2, 302);
    }
    else
    {
        vm_graphic_textout_to_layer(layer_hdl[0], 0, 262, w_no_result, g_no_result_len);
    }

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

    layer_hdl[0] = vm_graphic_create_layer(0, 0, screen_w, screen_h, -1);
    layer_buf = vm_graphic_get_layer_buffer(layer_hdl[0]);

    vm_reg_sysevt_callback(handle_sysevt);
    vm_switch_power_saving_mode(turn_off_mode);

    // [CHANGED] Single-channel (grayscale) 28x28 input for MNIST
    in.create(NN_W, NN_H, 1);

    init_number_lut();
    init_resize_lut();

    vm_ascii_to_ucs2(w_frame, 64, "Frame: ");
    vm_ascii_to_ucs2(w_nn, 64, "Ncnn: ");
    vm_ascii_to_ucs2(w_no_result, 64, "No result");

    g_frame_value_x = vm_graphic_get_string_width(w_frame);
    g_nn_value_x    = vm_graphic_get_string_width(w_nn);
    g_frame_len     = vm_wstrlen(w_frame);
    g_nn_len        = vm_wstrlen(w_nn);
    g_no_result_len = vm_wstrlen(w_no_result);

    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 1;

    g_classifier.opt = opt;

    // [CHANGED] Load MNIST int8 model binaries
    g_load_param_ret = g_classifier.load_param(mnist_int8_param_bin);
    g_load_model_ret = g_classifier.load_model(mnist_int8_bin);

    if (g_load_param_ret >= 0 && g_load_model_ret >= 0)
        g_classifier_loaded = 1;
    else
        g_classifier_loaded = 0;

    vm_create_camera_instance((VM_CAMERA_ID)vm_camera_get_main_camera_id(), &handle_c);

    vm_camera_register_notify(handle_c, cam_message_callback, 0);

    vm_cam_size_t sz = {240, 240};
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

void draw_top_class_line_ucs2(int rank, int top_index, int y)
{
    if (top_index < 0 || top_index >= 3)
        return;

    int cls = g_top_class[top_index];

    if (cls < 0 || cls >= 10)
        return;

    int percent = (int)(g_top_score[top_index] * 100.0f + 0.5f);

    if (percent < 0)   percent = 0;
    if (percent > 100) percent = 100;

    VMWCHAR line[32];
    int pos = 0;

    /* rank */
    if (rank == 1)      line[pos++] = '1';
    else if (rank == 2) line[pos++] = '2';
    else                line[pos++] = '3';

    line[pos++] = '.';
    line[pos++] = ' ';

    /* class name (single digit char for MNIST) */
    int name_len = vm_wstrlen((VMWCHAR*)class_names_w[cls]);

    for (int i = 0; i < name_len; i++)
        line[pos++] = class_names_w[cls][i];

    /* pad to fixed width (6 chars) */
    for (int i = name_len; i < 6; i++)
        line[pos++] = ' ';

    /* percent number */
    int num_len = vm_wstrlen(number_lut[percent]);

    for (int i = 0; i < num_len; i++)
        line[pos++] = number_lut[percent][i];

    line[pos++] = '%';
    line[pos]   = 0;

    vm_graphic_textout_to_layer(
        layer_hdl[0],
        0,
        y,
        line,
        vm_wstrlen(line)
    );
}
