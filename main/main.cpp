#include "vmsys.h"
#include "vmio.h"
#include "vmgraph.h"
#include "vmchset.h"
#include "vmstdlib.h"
#include "vm4res.h"
#include "vmres.h"
#include "vmpromng.h"

#include "yolo_cat_int8.id.h"
#include "yolo_cat_int8.mem.h"
//#include "yolo-fastestv2-opt.id.h"
//#include "yolo-fastestv2-opt.mem.h"

#include <net.h>
#include <cstdint>

VMINT		layer_hdl[1];	// layer handle array. 
VMUINT8* layer_buf = 0;

VMINT screen_w = 0;
VMINT screen_h = 0;
VMUINT8* demo_canvas = 0;

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
	
	vm_sprintf(str, "%d.%d%%", v / 10, v % 10);

	draw_text(l, x, y, str, c);
}

void find_cats() {
	ncnn::Option opt;
	opt.lightmode = true;
	opt.num_threads = 1;

	ncnn::Net yolo;
	yolo.opt = opt;

	yolo.load_param(cat_int8_param_bin);
	yolo.load_model(cat_int8_bin);

	ncnn::Mat in(96, 96, 3);

	float* b_ptr = in.channel(0);
	float* g_ptr = in.channel(1);
	float* r_ptr = in.channel(2);

	uint16_t* cat_rgb565 = (uint16_t*)(demo_canvas+32); 

	for (int y = 0; y < 96; y++) {
		for (int x = 0; x < 96; x++) {
			int src_x = (x * 100) / 96;
			int src_y = (y * 100) / 96;

			uint16_t pixel = cat_rgb565[src_y * 100 + src_x];

			uint8_t r5 = (pixel >> 11) & 0x1F;
			uint8_t g6 = (pixel >> 5) & 0x3F;
			uint8_t b5 = pixel & 0x1F;

			float R = (float)((r5 << 3) | (r5 >> 2));
			float G = (float)((g6 << 2) | (g6 >> 4));
			float B = (float)((b5 << 3) | (b5 >> 2));

			int out_idx = y * 96 + x;

			b_ptr[out_idx] = B * 0.00392156f;
			g_ptr[out_idx] = G * 0.00392156f;
			r_ptr[out_idx] = R * 0.00392156f;
		}
	}

	ncnn::Extractor ex = yolo.create_extractor();
	ex.set_light_mode(true);

	ex.input(cat_int8_param_id::LAYER_in0, in);

	ncnn::Mat out0, out1;

	ex.extract(cat_int8_param_id::BLOB_out0, out0);
	ex.extract(cat_int8_param_id::BLOB_out1, out1);

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

			int draw_x = (int)(detected_cats[i].x1 * (100.0f / 96.0f));
			int draw_y = (int)(detected_cats[i].y1 * (100.0f / 96.0f));
			int draw_w = (int)(cat_w * (100.0f / 96.0f));
			int draw_h = (int)(cat_h * (100.0f / 96.0f));

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

void vm_main(void) {
	layer_hdl[0] = -1;
	screen_w = vm_graphic_get_screen_width();
	screen_h = vm_graphic_get_screen_height();
	
	vm_reg_sysevt_callback(handle_sysevt);
	vm_reg_keyboard_callback(handle_keyevt);
	vm_reg_pen_callback(handle_penevt);

	VMINT size = 0;
	VMUINT8* res = vm_load_resource("demo.png", &size);
	demo_canvas = (VMUINT8*)vm_graphic_load_image(res, size);
	vm_free(res);

}

void handle_sysevt(VMINT message, VMINT param) {
#ifdef		SUPPORT_BG
	switch (message) {
	case VM_MSG_CREATE:
		break;
	case VM_MSG_PAINT:
		layer_hdl[0] = vm_graphic_create_layer(0, 0, screen_w, screen_h, -1);

		layer_buf = vm_graphic_get_layer_buffer(layer_hdl[0]);
		
		vm_graphic_set_clip(0, 0, screen_w, screen_h);
		
		draw_hello();
		break;
	case VM_MSG_HIDE:	
		if( layer_hdl[0] != -1 )
		{
			vm_graphic_delete_layer(layer_hdl[0]);
			layer_hdl[0] = -1;
		}
		break;
	case VM_MSG_QUIT:
		if( layer_hdl[0] != -1 )
		{
			vm_graphic_delete_layer(layer_hdl[0]);
			layer_hdl[0] = -1;
		}
		break;
	}
#else
	switch (message) {
	case VM_MSG_CREATE:
	case VM_MSG_ACTIVE:
		layer_hdl[0] = vm_graphic_create_layer(0, 0, screen_w, screen_h, -1);

		layer_buf = vm_graphic_get_layer_buffer(layer_hdl[0]);

		vm_graphic_set_clip(0, 0, screen_w, screen_h);
		break;
		
	case VM_MSG_PAINT:
		draw_hello();
		break;
		
	case VM_MSG_INACTIVE:
		if( layer_hdl[0] != -1 )
			vm_graphic_delete_layer(layer_hdl[0]);
		
		break;	
	case VM_MSG_QUIT:
		//if( layer_hdl[0] != -1 )
		//	vm_graphic_delete_layer(layer_hdl[0]);
		
		break;	
	}
#endif
}

void handle_keyevt(VMINT event, VMINT keycode) {}

void handle_penevt(VMINT event, VMINT x, VMINT y) {}

static void draw_hello(void) {
	vm_graphic_fill_rect(layer_buf, 0, 0, screen_w, screen_h, VM_COLOR_WHITE, VM_COLOR_WHITE);

	vm_graphic_blt(layer_buf, 0, 0, demo_canvas, 0, 0, 100, 100, 1);

	find_cats();

	vm_graphic_flush_layer(layer_hdl, 1);
}
