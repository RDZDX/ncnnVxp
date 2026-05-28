#include "vmsys.h"
#include "vmio.h"
#include "vmgraph.h"
#include "vmchset.h"
#include "vmstdlib.h"
#include "vm4res.h"
#include "vmres.h"
#include "vmpromng.h"
#include "vmcamera.h"

#define quantize 1

#ifdef quantize
#include "yolo_cat_int8.id.h"
#include "yolo_cat_int8.mem.h"

#define cat_param_bin cat_int8_param_bin
#define cat_bin cat_int8_bin
#define cat_param_id cat_int8_param_id
#else
#include "yolo_cat.id.h"
#include "yolo_cat.mem.h"
#endif

#include <net.h>
#include <cstdint>

VMINT		layer_hdl[1];	// layer handle array. 
VMUINT8* layer_buf = 0;

VMINT screen_w = 0;
VMINT screen_h = 0;
VMUINT8* demo_canvas = 0;
VM_CAMERA_HANDLE handle_c;

void handle_sysevt(VMINT message, VMINT param); // system events 
void handle_keyevt(VMINT event, VMINT keycode); // key events 
void handle_penevt(VMINT event, VMINT x, VMINT y); // pen events

#define MIN_VAL(a,b) ((a)<(b)?(a):(b))
#define MAX_VAL(a,b) ((a)>(b)?(a):(b))

struct CatBox {
	float x1, y1, x2, y2;
	float score;
	int keep;
};

float calculate_iou(CatBox a, CatBox b) {
	if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1) {
		return 0.0f;
	}

	float inter_width = MIN_VAL(a.x2, b.x2) - MAX_VAL(a.x1, b.x1);
	float inter_height = MIN_VAL(a.y2, b.y2) - MAX_VAL(a.y1, b.y1);
	float inter_area = inter_width * inter_height;

	float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
	float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);

	return inter_area / (area_a + area_b - inter_area);
}

static void draw_hello(void);

struct out_p {
	ncnn::Mat out1;
	ncnn::Mat out2;
};

void draw_text(int l, int x, int y, char* str, unsigned short c) {
	VMWCHAR wstr[100] = {};
	vm_ascii_to_ucs2(wstr, 200, str);

	vm_graphic_color color = { c };
	vm_graphic_setcolor(&color);

	vm_graphic_textout_to_layer(l, x, y, wstr, vm_wstrlen(wstr));
}

void draw_text_persents(int l, int x, int y, float p, unsigned short c) {
	char str[100] = {};

	int v = p * 10.f;

	sprintf(str, "%d.%d%%", v / 10, v % 10);

	draw_text(l, x, y, str, c);
}

void find_cats(ncnn::Mat &in) {
	ncnn::Option opt;
	opt.lightmode = true;
	opt.num_threads = 1;

	ncnn::Net yolo;
	yolo.opt = opt;

	yolo.load_param(cat_param_bin);
	yolo.load_model(cat_bin);

	ncnn::Extractor ex = yolo.create_extractor();
	ex.set_light_mode(true);

	ex.input(cat_param_id::LAYER_in0, in);

	ncnn::Mat out0, out1;

	ex.extract(cat_param_id::BLOB_out0, out0);
	ex.extract(cat_param_id::BLOB_out1, out1);

	const float anchor[12] = { 16.23f,26.11f, 28.11f,68.63f, 39.09f,40.95f,
							  49.31f,78.52f, 66.02f,54.64f, 79.19f,85.96f };

	const int MAX_CATS = 10;
	CatBox detected_cats[MAX_CATS];
	int cat_count = 0;

	ncnn::Mat outs[2] = { out0, out1 };

	for (int i = 0; i < 2; i++) {
		int outH = outs[i].c;
		int outW = outs[i].h;
		int outC = outs[i].w;
		int stride = 96 / outH;

		for (int y = 0; y < outH; y++) {
			const float* values = outs[i].channel(y);
			for (int x = 0; x < outW; x++) {
				for (int b = 0; b < 3; b++) {
					float objScore = values[12 + b];
					float clsScore = values[15];
					float final_prob = objScore * clsScore;

					if (final_prob > 0.01f && cat_count < MAX_CATS) {
						float bcx = ((values[b * 4 + 0] * 2.0f - 0.5f) + x) * stride;
						float bcy = ((values[b * 4 + 1] * 2.0f - 0.5f) + y) * stride;
						float bw = pow((values[b * 4 + 2] * 2.0f), 2.0f) * anchor[(i * 6) + b * 2 + 0];
						float bh = pow((values[b * 4 + 3] * 2.0f), 2.0f) * anchor[(i * 6) + b * 2 + 1];

						detected_cats[cat_count].x1 = bcx - 0.5f * bw;
						detected_cats[cat_count].y1 = bcy - 0.5f * bh;
						detected_cats[cat_count].x2 = bcx + 0.5f * bw;
						detected_cats[cat_count].y2 = bcy + 0.5f * bh;
						detected_cats[cat_count].score = final_prob;
						detected_cats[cat_count].keep = 1;

						cat_count++;
					}
				}
				values += outC;
			}
		}
	}

	for (int i = 0; i < cat_count - 1; i++) {
		for (int j = 0; j < cat_count - i - 1; j++) {
			if (detected_cats[j].score < detected_cats[j + 1].score) {
				CatBox temp = detected_cats[j];
				detected_cats[j] = detected_cats[j + 1];
				detected_cats[j + 1] = temp;
			}
		}
	}

	float nms_threshold = 0.45f;

	for (int i = 0; i < cat_count; i++) {
		if (detected_cats[i].keep == 0) continue;

		for (int j = i + 1; j < cat_count; j++) {
			if (detected_cats[j].keep == 1) {
				float iou = calculate_iou(detected_cats[i], detected_cats[j]);
				if (iou > nms_threshold) {
					detected_cats[j].keep = 0;
				}
			}
		}
	}

	const unsigned short cat_colors[10] = {
		VM_COLOR_888_TO_565(255, 0, 0),
		VM_COLOR_888_TO_565(0, 255, 0),
		VM_COLOR_888_TO_565(0, 0, 255),
		VM_COLOR_888_TO_565(255, 255, 0),
		VM_COLOR_888_TO_565(255, 0, 255),
		VM_COLOR_888_TO_565(0, 255, 255),
		VM_COLOR_888_TO_565(255, 128, 0),
		VM_COLOR_888_TO_565(128, 0, 255),
		VM_COLOR_888_TO_565(0, 255, 128),
		VM_COLOR_888_TO_565(255, 255, 255)
	};

	int actual_cats_drawn = 0;

	for (int i = 0; i < cat_count; i++) {
		if (detected_cats[i].keep == 1) {

			float cat_w = detected_cats[i].x2 - detected_cats[i].x1;
			float cat_h = detected_cats[i].y2 - detected_cats[i].y1;

			int draw_x = (int)(detected_cats[i].x1 * (240.0f / 96.0f));
			int draw_y = (int)(detected_cats[i].y1 * (240.0f / 96.0f));
			int draw_w = (int)(cat_w * (240.0f / 96.0f));
			int draw_h = (int)(cat_h * (240.0f / 96.0f));

			if (draw_x < 0) draw_x = 0;
			if (draw_y < 0) draw_y = 0;

			unsigned short current_color = cat_colors[i % 10];

			if (draw_w > 0 && draw_h > 0) {
				vm_graphic_rect(layer_buf, draw_x, draw_y, draw_w, draw_h, current_color);
				vm_graphic_rect(layer_buf, draw_x + 1, draw_y + 1, draw_w - 2, draw_h - 2, current_color);
				draw_text_persents(layer_hdl[0], draw_x + 1, draw_y + 1, detected_cats[i].score * 100.f, current_color);

				actual_cats_drawn++;
			}
		}
	}
}

static float float_lut[256];
void init_lut() {
	for (int i = 0; i < 256; i++) {
		float_lut[i] = (float)i * 0.003921568f;
	}
}

unsigned char claim(int x) {
	return x > 255 ? 255 : (x < 0 ? 0 : x);
}

void cam_message_callback(const vm_cam_notify_data_t* notify_data, void* user_data) {
	if (notify_data != NULL)
	{
		vm_cam_frame_data_t frame;

		switch (notify_data->cam_message) {
		case VM_CAM_PREVIEW_FRAME_RECEIVED:
			if (vm_camera_get_frame(handle_c, &frame) == VM_CAM_SUCCESS)
			{
				unsigned int time1 = vm_get_tick_count();

				ncnn::Mat in(96, 96, 3);

				float* ptr_r = in.channel(0);
				float* ptr_g = in.channel(1);
				float* ptr_b = in.channel(2);

				//int idx = 0;
				int i = 0;

				int nn_x = 0;
				int nn_y = 0;

				VMUINT app_frame_data_size = 0;
				VMUINT8* app_frame_data = NULL;
				{
					unsigned short* layer_buf_s = (unsigned short*)layer_buf;
					unsigned char* s = (unsigned char*)frame.pixtel_data;
					for (unsigned int y = 0; y < 240; ++y)
						for (unsigned int x = 0; x < 240; x += 2)
						{
							int u0 = *s++ - 128;
							int y0 = *s++ - 16;
							int v = *s++ - 128;
							int y2 = *s++ - 16;

							int tmp = 298 * y0 + 128;
							int vv = 409 * v;
							int uv = -100 * u0 - 208 * v;
							int uu = 516 * u0;

							unsigned char r = claim((tmp + vv) >> 8);
							unsigned char g = claim((tmp + uv) >> 8);
							unsigned char b = claim((tmp + uu) >> 8);

							layer_buf_s[i] = VM_COLOR_888_TO_565(r, g, b);

							if ((x * 96 / 240 == nn_x) && (y * 96 / 240 == nn_y)) {
								int idx = nn_y * 96 + nn_x;
								ptr_r[idx] = float_lut[r];
								ptr_g[idx] = float_lut[g];
								ptr_b[idx] = float_lut[b];

								nn_x++;
								if (nn_x >= 96) { nn_x = 0; nn_y++; }
							}

							tmp = 298 * y2 + 128;
							r = claim((tmp + vv) >> 8);
							g = claim((tmp + uv) >> 8);
							b = claim((tmp + uu) >> 8);

							layer_buf_s[i + 1] = VM_COLOR_888_TO_565(r, g, b);

							if (((x+1) * 96 / 240 == nn_x) && (y * 96 / 240 == nn_y)) {
								int idx = nn_y * 96 + nn_x;
								ptr_r[idx] = float_lut[r];
								ptr_g[idx] = float_lut[g];
								ptr_b[idx] = float_lut[b];

								nn_x++;
								if (nn_x >= 96) { nn_x = 0; nn_y++; }
							}

							i += 2;
						}
				}

				/*unsigned short* layer_buf_s = (unsigned short*)layer_buf;

				for (int y = 0; y < 96; ++y)
					for (int x = 0; x < 96; ++x) {
						int idx = y*96+x;
						unsigned short r = (unsigned short)(ptr_r[idx] * 255.0f);
						unsigned short g = (unsigned short)(ptr_g[idx] * 255.0f);
						unsigned short b = (unsigned short)(ptr_b[idx] * 255.0f);

						layer_buf_s[y * 240 + x] = VM_COLOR_888_TO_565(r, g, b);;
					}*/


				vm_graphic_fill_rect(layer_buf, 0, 240, screen_w, screen_h - 240, VM_COLOR_WHITE, VM_COLOR_WHITE);

				unsigned int time2 = vm_get_tick_count();
				find_cats(in);
				unsigned int time3 = vm_get_tick_count();

				char str[100] = {};
				sprintf(str, "Frame processing: %d ms", time2 - time1);
				draw_text(layer_hdl[0], 0, 240, str, 0x0000);
				sprintf(str, "ncnn computing: %d ms", time3 - time2);
				draw_text(layer_hdl[0], 0, 240 + vm_graphic_get_character_height(), str, 0x0000);

				vm_graphic_flush_layer(layer_hdl, 1);

			}
			break;
		}
	}
}

void vm_main(void) {
	layer_hdl[0] = -1;
	screen_w = vm_graphic_get_screen_width();
	screen_h = vm_graphic_get_screen_height();

	layer_hdl[0] = vm_graphic_create_layer(0, 0, screen_w, screen_h, -1);
	layer_buf = vm_graphic_get_layer_buffer(layer_hdl[0]);
	vm_graphic_set_clip(0, 0, screen_w, screen_h);

	vm_reg_sysevt_callback(handle_sysevt);
	vm_reg_keyboard_callback(handle_keyevt);
	vm_reg_pen_callback(handle_penevt);

	vm_switch_power_saving_mode(turn_off_mode);

	init_lut();

	vm_create_camera_instance((VM_CAMERA_ID)vm_camera_get_main_camera_id(), &handle_c);
	vm_camera_register_notify(handle_c, cam_message_callback, 0);

	vm_cam_size_t my_cam_size;
	my_cam_size.width = 240;
	my_cam_size.height = 240;
	vm_camera_set_preview_size(handle_c, &my_cam_size);

	vm_camera_preview_start(handle_c);
}

void handle_sysevt(VMINT message, VMINT param) {
#ifdef		SUPPORT_BG
	switch (message) {
	case VM_MSG_CREATE:
		break;
	case VM_MSG_PAINT:
		break;
	case VM_MSG_HIDE:
		break;
	case VM_MSG_QUIT:
		vm_release_camera_instance(handle_c);
		break;
	}
#else
	switch (message) {
	case VM_MSG_CREATE:
	case VM_MSG_ACTIVE:
		break;

	case VM_MSG_PAINT:
		break;

	case VM_MSG_INACTIVE:
		break;
	case VM_MSG_QUIT:
		vm_release_camera_instance(handle_c);
		break;
	}
#endif
}

void handle_keyevt(VMINT event, VMINT keycode) {}

void handle_penevt(VMINT event, VMINT x, VMINT y) {}

static void draw_hello(void) {
	vm_graphic_fill_rect(layer_buf, 0, 0, screen_w, screen_h, VM_COLOR_WHITE, VM_COLOR_WHITE);

	vm_graphic_blt(layer_buf, 0, 0, demo_canvas, 0, 0, 100, 100, 1);

	//find_cats();

	vm_graphic_flush_layer(layer_hdl, 1);
}
