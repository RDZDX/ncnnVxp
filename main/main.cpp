#include "vmsys.h"
#include "vmgraph.h"
#include "vmchset.h"
#include "vmstdlib.h"
#include "vmcamera.h"
#include <net.h>

#include "face_detection_yunet_2026may-opt.id.h"
#include "face_detection_yunet_2026may-opt.mem.h"

static ncnn::Net g_yunet;
static int g_yunet_loaded = 0;

VMINT layer_hdl[1];
VMUINT8* layer_buf = 0;
VMINT screen_w = 0;
VMINT screen_h = 0;
VM_CAMERA_HANDLE handle_c;

VMWCHAR w_frame[64];
VMWCHAR w_nn[64];
VMWCHAR number_lut[1000][5];
static int g_frame_len;
static int g_nn_len;
static int g_frame_value_x;
static int g_nn_value_x;

static unsigned char src_x[96];
static unsigned char src_y[96];

static ncnn::Mat in;

struct CatBox {
    float x1, y1, x2, y2;
    float score;
    int keep;
};

struct CatBox detected_cats[100];
int cat_count_global = 0;

void handle_sysevt(VMINT message, VMINT param);
void init_resize_lut();
void init_number_lut();

#define MIN_VAL(a,b) ((a)<(b)?(a):(b))
#define MAX_VAL(a,b) ((a)>(b)?(a):(b))

float calculate_iou(CatBox a, CatBox b)
{
    if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1)
        return 0.0f;

    float iw = MIN_VAL(a.x2, b.x2) - MAX_VAL(a.x1, b.x1);
    float ih = MIN_VAL(a.y2, b.y2) - MAX_VAL(a.y1, b.y1);

    float inter = iw * ih;

    float aa = (a.x2 - a.x1) * (a.y2 - a.y1);
    float bb = (b.x2 - b.x1) * (b.y2 - b.y1);

    return inter / (aa + bb - inter);
}

static inline unsigned char claim(int x)
{
    return (unsigned char)(x > 255 ? 255 : (x < 0 ? 0 : x));
}

void nms_boxes(CatBox* boxes, int& count, float nms_threshold)
{
    for (int i = 0; i < count; i++)
    {
        if (!boxes[i].keep) continue;
        for (int j = i + 1; j < count; j++)
        {
            if (!boxes[j].keep) continue;
            float iou = calculate_iou(boxes[i], boxes[j]);
            if (iou > nms_threshold)
            {
                if (boxes[i].score > boxes[j].score)
                    boxes[j].keep = 0;
                else
                    boxes[i].keep = 0;
            }
        }
    }

    int idx = 0;
    for (int i = 0; i < count; i++)
    {
        if (boxes[i].keep)
        {
            if (idx != i) boxes[idx] = boxes[i];
            idx++;
        }
    }
    count = idx;
}

void find_cats(ncnn::Mat& in)
{
    if (!g_yunet_loaded)
        return;

    cat_count_global = 0;

    ncnn::Extractor ex = g_yunet.create_extractor();
    ex.set_light_mode(true);
    ex.input(face_detection_yunet_2026may_opt_param_id::BLOB_in0, in);

    ncnn::Mat cls_8,  cls_16,  cls_32;
    ncnn::Mat obj_8,  obj_16,  obj_32;
    ncnn::Mat bbox_8, bbox_16, bbox_32;

    ex.extract(face_detection_yunet_2026may_opt_param_id::BLOB_out0, cls_8);
    ex.extract(face_detection_yunet_2026may_opt_param_id::BLOB_out1, cls_16);
    ex.extract(face_detection_yunet_2026may_opt_param_id::BLOB_out2, cls_32);
    ex.extract(face_detection_yunet_2026may_opt_param_id::BLOB_out3, obj_8);
    ex.extract(face_detection_yunet_2026may_opt_param_id::BLOB_out4, obj_16);
    ex.extract(face_detection_yunet_2026may_opt_param_id::BLOB_out5, obj_32);
    ex.extract(face_detection_yunet_2026may_opt_param_id::BLOB_out6, bbox_8);
    ex.extract(face_detection_yunet_2026may_opt_param_id::BLOB_out7, bbox_16);
    ex.extract(face_detection_yunet_2026may_opt_param_id::BLOB_out8, bbox_32);

    const float conf_threshold = 0.6f;
    const float nms_threshold  = 0.45f;

    struct ScaleInfo {
        ncnn::Mat* cls;
        ncnn::Mat* obj;
        ncnn::Mat* bbox;
        int stride;
        int gw;
    } scales[3] = {
        { &cls_8,  &obj_8,  &bbox_8,   8, 12 },
        { &cls_16, &obj_16, &bbox_16, 16,  6 },
        { &cls_32, &obj_32, &bbox_32, 32,  3 },
    };

    for (int s = 0; s < 3; s++)
    {
        ncnn::Mat& cls  = *scales[s].cls;
        ncnn::Mat& obj  = *scales[s].obj;
        ncnn::Mat& bbox = *scales[s].bbox;
        int stride = scales[s].stride;
        int gw     = scales[s].gw;

        const float* cls_ptr = (const float*)cls.data;
        const float* obj_ptr = (const float*)obj.data;
        int num_anchors = cls.h;

        for (int i = 0; i < num_anchors && cat_count_global < 100; i++)
        {
            float score = cls_ptr[i] * obj_ptr[i];
            if (score < conf_threshold)
                continue;

            const float* row = bbox.row(i);
            int grid_x = i % gw;
            int grid_y = i / gw;

            float cx = (grid_x + row[0]) * stride;
            float cy = (grid_y + row[1]) * stride;
            float bw = expf(row[2]) * stride;
            float bh = expf(row[3]) * stride;

            detected_cats[cat_count_global].x1    = cx - bw * 0.5f;
            detected_cats[cat_count_global].y1    = cy - bh * 0.5f;
            detected_cats[cat_count_global].x2    = cx + bw * 0.5f;
            detected_cats[cat_count_global].y2    = cy + bh * 0.5f;
            detected_cats[cat_count_global].score = score;
            detected_cats[cat_count_global].keep  = 1;
            cat_count_global++;
        }
    }

    if (cat_count_global > 0)
        nms_boxes(detected_cats, cat_count_global, nms_threshold);
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

    /* Convert YUV -> RGB565, draw camera image */
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

    /* Build 96x96 NN input */
    unsigned short* fb = (unsigned short*)layer_buf;

    for (int ny = 0; ny < 96; ny++)
    {
        int sy = src_y[ny];
        float* rowB = (float*)in.channel(0) + ny * in.w;
        float* rowG = (float*)in.channel(1) + ny * in.w;
        float* rowR = (float*)in.channel(2) + ny * in.w;

        for (int nx = 0; nx < 96; nx++)
        {
            unsigned short p = fb[sy * 240 + src_x[nx]];
            unsigned char r = ((p >> 11) & 0x1F) << 3;
            unsigned char g = ((p >> 5)  & 0x3F) << 2;
            unsigned char b = (p & 0x1F) << 3;

            rowB[nx] = (float)b - 104.0f;
            rowG[nx] = (float)g - 117.0f;
            rowR[nx] = (float)r - 123.0f;
        }
    }

    unsigned int time2 = vm_get_tick_count();

    find_cats(in);

    unsigned int time3 = vm_get_tick_count();

    /* Draw detections */
    const unsigned short box_color = VM_COLOR_888_TO_565(255, 0, 0);
    const float scale = 240.0f / 96.0f;

    for (int i = 0; i < cat_count_global; i++)
    {
        if (!detected_cats[i].keep) continue;

        int dx = (int)(detected_cats[i].x1 * scale);
        int dy = (int)(detected_cats[i].y1 * scale);
        int dw = (int)((detected_cats[i].x2 - detected_cats[i].x1) * scale);
        int dh = (int)((detected_cats[i].y2 - detected_cats[i].y1) * scale);

        if (dx < 0) dx = 0;
        if (dy < 0) dy = 0;
        if (dx + dw > 240) dw = 240 - dx;
        if (dy + dh > 240) dh = 240 - dy;

        if (dw <= 0 || dh <= 0) continue;

        vm_graphic_rect(layer_buf, dx, dy, dw, dh, box_color);
    }

    /* Timing display */
    vm_graphic_fill_rect(layer_buf, 0, 240, screen_w, screen_h - 240, VM_COLOR_WHITE, VM_COLOR_WHITE);

    vm_graphic_color color = {VM_COLOR_888_TO_565(0, 0, 0)};
    vm_graphic_setcolor(&color);

    int frame_ms = (int)(time2 - time1);
    if (frame_ms > 999) frame_ms = 999;

    vm_graphic_textout_to_layer(layer_hdl[0], 0, 241, w_frame, g_frame_len);
    vm_graphic_textout_to_layer(layer_hdl[0], g_frame_value_x, 241, number_lut[frame_ms], vm_wstrlen(number_lut[frame_ms]));

    int nn_ms = (int)(time3 - time2);
    if (nn_ms > 9999) nn_ms = 9999;

    vm_graphic_textout_to_layer(layer_hdl[0], 0, 260, w_nn, g_nn_len);

    {
        VMWCHAR ms_str[8];
        int v = nn_ms;
        int pos = 0;
        if (v >= 1000) ms_str[pos++] = '0' + v / 1000;
        if (v >= 100)  ms_str[pos++] = '0' + (v / 100) % 10;
        if (v >= 10)   ms_str[pos++] = '0' + (v / 10) % 10;
                       ms_str[pos++] = '0' + v % 10;
        ms_str[pos] = 0;
        vm_graphic_textout_to_layer(layer_hdl[0], g_nn_value_x, 260, ms_str, pos);
    }

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

    in.create(96, 96, 3);

    init_number_lut();
    init_resize_lut();

    vm_ascii_to_ucs2(w_frame, 64, "Frame processing ms: ");
    vm_ascii_to_ucs2(w_nn, 64, "Ncnn computing ms: ");

    g_frame_value_x = vm_graphic_get_string_width(w_frame);
    g_nn_value_x    = vm_graphic_get_string_width(w_nn);
    g_frame_len     = vm_wstrlen(w_frame);
    g_nn_len        = vm_wstrlen(w_nn);

    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 1;
    g_yunet.opt = opt;

    g_yunet.load_param(face_detection_yunet_2026may_opt_param_bin);
    g_yunet.load_model(face_detection_yunet_2026may_opt_bin);
    g_yunet_loaded = 1;

    vm_create_camera_instance((VM_CAMERA_ID)vm_camera_get_main_camera_id(), &handle_c);
    vm_camera_register_notify(handle_c, cam_message_callback, 0);

    vm_cam_size_t sz = {240, 240};
    vm_camera_set_preview_size(handle_c, &sz);
    vm_camera_preview_start(handle_c);
}

void handle_sysevt(VMINT message, VMINT param)
{
    switch (message)
    {
    case VM_MSG_QUIT:
        g_yunet.clear();
        vm_release_camera_instance(handle_c);
        break;
    default:
        break;
    }
}

void init_resize_lut()
{
    for (int i = 0; i < 96; i++)
    {
        src_x[i] = (unsigned char)((i * 240) / 96);
        src_y[i] = (unsigned char)((i * 240) / 96);
    }
}

void init_number_lut()
{
    for (int n = 0; n < 1000; n++)
    {
        int pos = 0;
        if (n >= 100) { number_lut[n][pos++] = '0' + (n / 100);         }
        if (n >= 10)  { number_lut[n][pos++] = '0' + ((n / 10) % 10);   }
                        number_lut[n][pos++] = '0' + (n % 10);
                        number_lut[n][pos]   = 0;
    }
}
