#!/usr/bin/env python3
import math
import os
import cairo  # type: ignore[import-not-found]

OUTPUT_DIR = '/home/murilods/projetos/semaforo/assets'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def rounded_rect(ctx, x, y, w, h, r):
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi/2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi/2)
    ctx.arc(x + r, y + h - r, r, math.pi/2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3*math.pi/2)
    ctx.close_path()

def draw_shadow(ctx, x, y, w, h, r):
    ctx.save()
    rounded_rect(ctx, x - 2, y + 4, w + 4, h + 4, r + 2)
    ctx.set_source_rgba(0, 0, 0, 0.35)
    ctx.fill()
    ctx.restore()

def draw_wheels(ctx, x, y, w, h, front_offset, rear_offset, wheel_w=8, wheel_h=26):
    ctx.save()
    ctx.set_source_rgba(0.1, 0.1, 0.12, 0.95)
    pts = [
        (x - 3, y + front_offset),
        (x + w - wheel_w + 3, y + front_offset),
        (x - 3, y + h - rear_offset - wheel_h),
        (x + w - wheel_w + 3, y + h - rear_offset - wheel_h)
    ]
    for wx, wy in pts:
        rounded_rect(ctx, wx, wy, wheel_w, wheel_h, 3)
        ctx.fill()
    ctx.restore()

def generate_car_sedan(filename='car_sedan.png'):
    w, h = 120, 240
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(surface)
    cx = w / 2
    car_w, car_h = 76, 196
    car_x, car_y = cx - car_w / 2, 22
    r = 16
    draw_shadow(ctx, car_x, car_y, car_w, car_h, r)
    draw_wheels(ctx, car_x, car_y, car_w, car_h, 34, 30, 8, 26)
    
    ctx.save()
    rounded_rect(ctx, car_x, car_y, car_w, car_h, r)
    grad = cairo.LinearGradient(car_x, car_y, car_x + car_w, car_y)
    grad.add_color_stop_rgb(0.0, 0.08, 0.22, 0.55)
    grad.add_color_stop_rgb(0.2, 0.14, 0.38, 0.85)
    grad.add_color_stop_rgb(0.5, 0.22, 0.50, 0.98)
    grad.add_color_stop_rgb(0.8, 0.14, 0.38, 0.85)
    grad.add_color_stop_rgb(1.0, 0.08, 0.22, 0.55)
    ctx.set_source(grad); ctx.fill_preserve()
    ctx.set_source_rgba(0.04, 0.12, 0.30, 0.9); ctx.set_line_width(2); ctx.stroke(); ctx.restore()
    
    # Retrovisores
    ctx.save(); ctx.set_source_rgb(0.12, 0.32, 0.75)
    rounded_rect(ctx, car_x - 7, car_y + 54, 8, 14, 3); ctx.fill()
    rounded_rect(ctx, car_x + car_w - 1, car_y + 54, 8, 14, 3); ctx.fill(); ctx.restore()
    
    # Vidros
    cab_x, cab_y, cab_w, cab_h = car_x + 7, car_y + 50, car_w - 14, 98
    ctx.save(); rounded_rect(ctx, cab_x, cab_y, cab_w, cab_h, 10); ctx.set_source_rgb(0.10, 0.12, 0.16); ctx.fill()
    rounded_rect(ctx, cab_x + 3, cab_y + 3, cab_w - 6, 26, 6); ctx.set_source_rgb(0.20, 0.35, 0.45); ctx.fill()
    rounded_rect(ctx, cab_x + 5, cab_y + 32, cab_w - 10, 42, 5); ctx.set_source_rgb(0.18, 0.40, 0.85); ctx.fill()
    rounded_rect(ctx, cab_x + 4, cab_y + 75, cab_w - 8, 20, 5); ctx.set_source_rgb(0.15, 0.25, 0.35); ctx.fill(); ctx.restore()
    
    ctx.save(); ctx.set_source_rgb(1.0, 0.98, 0.85)
    rounded_rect(ctx, car_x + 6, car_y + 4, 12, 10, 3); ctx.fill()
    rounded_rect(ctx, car_x + car_w - 18, car_y + 4, 12, 10, 3); ctx.fill()
    ctx.set_source_rgb(0.85, 0.08, 0.08)
    rounded_rect(ctx, car_x + 6, car_y + car_h - 9, 14, 6, 2); ctx.fill()
    rounded_rect(ctx, car_x + car_w - 20, car_y + car_h - 9, 14, 6, 2); ctx.fill(); ctx.restore()
    surface.write_to_png(os.path.join(OUTPUT_DIR, filename))

def generate_car_hatch(filename='car_hatch.png'):
    w, h = 110, 210
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(surface)
    cx = w / 2
    car_w, car_h = 72, 174
    car_x, car_y = cx - car_w / 2, 18
    r = 15
    draw_shadow(ctx, car_x, car_y, car_w, car_h, r)
    draw_wheels(ctx, car_x, car_y, car_w, car_h, 28, 22, 8, 24)
    
    ctx.save(); rounded_rect(ctx, car_x, car_y, car_w, car_h, r)
    grad = cairo.LinearGradient(car_x, car_y, car_x + car_w, car_y)
    grad.add_color_stop_rgb(0.0, 0.60, 0.06, 0.06)
    grad.add_color_stop_rgb(0.5, 0.95, 0.25, 0.25)
    grad.add_color_stop_rgb(1.0, 0.60, 0.06, 0.06)
    ctx.set_source(grad); ctx.fill_preserve()
    ctx.set_source_rgba(0.35, 0.02, 0.02, 0.9); ctx.set_line_width(1.8); ctx.stroke(); ctx.restore()
    
    cab_x, cab_y, cab_w, cab_h = car_x + 6, car_y + 42, car_w - 12, 94
    ctx.save(); rounded_rect(ctx, cab_x, cab_y, cab_w, cab_h, 8); ctx.set_source_rgb(0.12, 0.14, 0.16); ctx.fill()
    rounded_rect(ctx, cab_x + 3, cab_y + 3, cab_w - 6, 24, 5); ctx.set_source_rgb(0.20, 0.35, 0.45); ctx.fill()
    rounded_rect(ctx, cab_x + 4, cab_y + 30, cab_w - 8, 44, 4); ctx.set_source_rgb(0.85, 0.15, 0.15); ctx.fill()
    rounded_rect(ctx, cab_x + 4, cab_y + 76, cab_w - 8, 16, 4); ctx.set_source_rgb(0.16, 0.25, 0.35); ctx.fill()
    ctx.set_source_rgb(0.15, 0.15, 0.15); rounded_rect(ctx, car_x + 10, car_y + car_h - 7, car_w - 20, 5, 2); ctx.fill(); ctx.restore()
    
    ctx.save(); ctx.set_source_rgb(0.98, 0.98, 0.85)
    rounded_rect(ctx, car_x + 5, car_y + 4, 11, 8, 2); ctx.fill()
    rounded_rect(ctx, car_x + car_w - 16, car_y + 4, 11, 8, 2); ctx.fill()
    ctx.set_source_rgb(0.90, 0.10, 0.10)
    rounded_rect(ctx, car_x + 5, car_y + car_h - 10, 10, 6, 2); ctx.fill()
    rounded_rect(ctx, car_x + car_w - 15, car_y + car_h - 10, 10, 6, 2); ctx.fill(); ctx.restore()
    surface.write_to_png(os.path.join(OUTPUT_DIR, filename))

def generate_car_sports(filename='car_sports.png'):
    w, h = 120, 240
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(surface)
    cx = w / 2
    car_w, car_h = 78, 202
    car_x, car_y = cx - car_w / 2, 19
    draw_shadow(ctx, car_x, car_y, car_w, car_h, 16)
    draw_wheels(ctx, car_x, car_y, car_w, car_h, 32, 28, 9, 28)
    ctx.save(); rounded_rect(ctx, car_x, car_y, car_w, car_h, 16)
    grad = cairo.LinearGradient(car_x, car_y, car_x + car_w, car_y)
    grad.add_color_stop_rgb(0.0, 0.10, 0.10, 0.12)
    grad.add_color_stop_rgb(0.5, 0.28, 0.28, 0.32)
    grad.add_color_stop_rgb(1.0, 0.10, 0.10, 0.12)
    ctx.set_source(grad); ctx.fill_preserve()
    ctx.set_source_rgb(0.05, 0.05, 0.06); ctx.set_line_width(2); ctx.stroke(); ctx.restore()
    
    # Listras esportivas
    ctx.save(); ctx.set_source_rgb(0.95, 0.65, 0.10)
    ctx.rectangle(cx - 8, car_y + 4, 5, car_h - 12); ctx.fill()
    ctx.rectangle(cx + 3, car_y + 4, 5, car_h - 12); ctx.fill(); ctx.restore()
    
    cab_x, cab_y, cab_w, cab_h = car_x + 8, car_y + 54, car_w - 16, 92
    ctx.save(); rounded_rect(ctx, cab_x, cab_y, cab_w, cab_h, 10); ctx.set_source_rgb(0.08, 0.08, 0.10); ctx.fill()
    rounded_rect(ctx, cab_x + 3, cab_y + 4, cab_w - 6, 28, 6); ctx.set_source_rgb(0.18, 0.30, 0.40); ctx.fill()
    rounded_rect(ctx, cab_x + 4, cab_y + 34, cab_w - 8, 36, 4); ctx.set_source_rgb(0.20, 0.20, 0.24); ctx.fill()
    rounded_rect(ctx, cab_x + 3, cab_y + 72, cab_w - 6, 18, 4); ctx.set_source_rgb(0.16, 0.24, 0.32); ctx.fill()
    rounded_rect(ctx, car_x + 4, car_y + car_h - 12, car_w - 8, 8, 3); ctx.set_source_rgb(0.12, 0.12, 0.14); ctx.fill(); ctx.restore()
    
    ctx.save(); ctx.set_source_rgb(0.85, 0.95, 1.0)
    rounded_rect(ctx, car_x + 5, car_y + 4, 14, 8, 3); ctx.fill()
    rounded_rect(ctx, car_x + car_w - 19, car_y + 4, 14, 8, 3); ctx.fill()
    ctx.set_source_rgb(0.95, 0.05, 0.05)
    rounded_rect(ctx, car_x + 5, car_y + car_h - 6, 14, 4, 1); ctx.fill()
    rounded_rect(ctx, car_x + car_w - 19, car_y + car_h - 6, 14, 4, 1); ctx.fill(); ctx.restore()
    surface.write_to_png(os.path.join(OUTPUT_DIR, filename))

def generate_taxi(filename='taxi.png'):
    w, h = 120, 240
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(surface)
    cx = w / 2
    car_w, car_h = 76, 196
    car_x, car_y = cx - car_w / 2, 22
    draw_shadow(ctx, car_x, car_y, car_w, car_h, 16)
    draw_wheels(ctx, car_x, car_y, car_w, car_h, 34, 30, 8, 26)
    ctx.save(); rounded_rect(ctx, car_x, car_y, car_w, car_h, 16)
    grad = cairo.LinearGradient(car_x, car_y, car_x + car_w, car_y)
    grad.add_color_stop_rgb(0.0, 0.85, 0.70, 0.05)
    grad.add_color_stop_rgb(0.5, 0.98, 0.85, 0.15)
    grad.add_color_stop_rgb(1.0, 0.85, 0.70, 0.05)
    ctx.set_source(grad); ctx.fill_preserve()
    ctx.set_source_rgb(0.6, 0.45, 0.0); ctx.set_line_width(2); ctx.stroke(); ctx.restore()
    
    cab_x, cab_y, cab_w, cab_h = car_x + 7, car_y + 50, car_w - 14, 98
    ctx.save(); rounded_rect(ctx, cab_x, cab_y, cab_w, cab_h, 10); ctx.set_source_rgb(0.12, 0.12, 0.14); ctx.fill()
    rounded_rect(ctx, cab_x + 3, cab_y + 3, cab_w - 6, 26, 6); ctx.set_source_rgb(0.20, 0.35, 0.45); ctx.fill()
    rounded_rect(ctx, cab_x + 4, cab_y + 32, cab_w - 8, 42, 5); ctx.set_source_rgb(0.95, 0.82, 0.10); ctx.fill()
    rounded_rect(ctx, cab_x + 4, cab_y + 75, cab_w - 8, 20, 5); ctx.set_source_rgb(0.18, 0.30, 0.40); ctx.fill()
    rounded_rect(ctx, cx - 14, cab_y + 46, 28, 12, 3); ctx.set_source_rgb(0.98, 0.98, 0.98); ctx.fill_preserve()
    ctx.set_source_rgb(0.2, 0.2, 0.2); ctx.set_line_width(1); ctx.stroke(); ctx.restore()
    
    ctx.save(); ctx.set_source_rgb(1.0, 0.98, 0.85)
    rounded_rect(ctx, car_x + 6, car_y + 4, 12, 10, 3); ctx.fill()
    rounded_rect(ctx, car_x + car_w - 18, car_y + 4, 12, 10, 3); ctx.fill()
    ctx.set_source_rgb(0.85, 0.08, 0.08)
    rounded_rect(ctx, car_x + 6, car_y + car_h - 9, 14, 6, 2); ctx.fill()
    rounded_rect(ctx, car_x + car_w - 20, car_y + car_h - 9, 14, 6, 2); ctx.fill(); ctx.restore()
    surface.write_to_png(os.path.join(OUTPUT_DIR, filename))

def generate_police(filename='police.png'):
    w, h = 120, 240
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(surface)
    cx = w / 2
    car_w, car_h = 76, 196
    car_x, car_y = cx - car_w / 2, 22
    draw_shadow(ctx, car_x, car_y, car_w, car_h, 16)
    draw_wheels(ctx, car_x, car_y, car_w, car_h, 34, 30, 8, 26)
    ctx.save(); rounded_rect(ctx, car_x, car_y, car_w, car_h, 16)
    ctx.set_source_rgb(0.10, 0.10, 0.12); ctx.fill_preserve()
    ctx.set_source_rgb(0.04, 0.04, 0.05); ctx.set_line_width(2); ctx.stroke()
    ctx.set_source_rgb(0.95, 0.95, 0.98); ctx.rectangle(car_x + 6, car_y + 48, car_w - 12, 98); ctx.fill(); ctx.restore()
    
    cab_x, cab_y, cab_w, cab_h = car_x + 7, car_y + 50, car_w - 14, 98
    ctx.save(); rounded_rect(ctx, cab_x, cab_y, cab_w, cab_h, 10); ctx.set_source_rgb(0.10, 0.12, 0.16); ctx.fill()
    rounded_rect(ctx, cab_x + 3, cab_y + 3, cab_w - 6, 26, 6); ctx.set_source_rgb(0.20, 0.35, 0.45); ctx.fill()
    rounded_rect(ctx, cab_x + 4, cab_y + 32, cab_w - 8, 42, 5); ctx.set_source_rgb(0.95, 0.95, 0.98); ctx.fill()
    rounded_rect(ctx, cab_x + 4, cab_y + 75, cab_w - 8, 20, 5); ctx.set_source_rgb(0.18, 0.30, 0.40); ctx.fill()
    # Giroflex Policial
    rounded_rect(ctx, cx - 18, cab_y + 46, 16, 12, 3); ctx.set_source_rgb(0.95, 0.1, 0.1); ctx.fill()
    rounded_rect(ctx, cx + 2, cab_y + 46, 16, 12, 3); ctx.set_source_rgb(0.1, 0.4, 0.95); ctx.fill(); ctx.restore()
    
    ctx.save(); ctx.set_source_rgb(1.0, 0.98, 0.85)
    rounded_rect(ctx, car_x + 6, car_y + 4, 12, 10, 3); ctx.fill()
    rounded_rect(ctx, car_x + car_w - 18, car_y + 4, 12, 10, 3); ctx.fill()
    ctx.set_source_rgb(0.85, 0.08, 0.08)
    rounded_rect(ctx, car_x + 6, car_y + car_h - 9, 14, 6, 2); ctx.fill()
    rounded_rect(ctx, car_x + car_w - 20, car_y + car_h - 9, 14, 6, 2); ctx.fill(); ctx.restore()
    surface.write_to_png(os.path.join(OUTPUT_DIR, filename))

def generate_ambulance(filename='ambulance.png'):
    w, h = 120, 250
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(surface)
    cx = w / 2
    car_w, car_h = 78, 212
    car_x, car_y = cx - car_w / 2, 20
    draw_shadow(ctx, car_x, car_y, car_w, car_h, 14)
    draw_wheels(ctx, car_x, car_y, car_w, car_h, 34, 32, 8, 26)
    ctx.save(); rounded_rect(ctx, car_x, car_y, car_w, car_h, 14)
    ctx.set_source_rgb(0.95, 0.95, 0.97); ctx.fill_preserve()
    ctx.set_source_rgb(0.70, 0.72, 0.76); ctx.set_line_width(2); ctx.stroke(); ctx.restore()
    
    ctx.save(); ctx.set_source_rgb(0.85, 0.12, 0.12)
    ctx.rectangle(car_x + 3, car_y + 40, 5, car_h - 60); ctx.fill()
    ctx.rectangle(car_x + car_w - 8, car_y + 40, 5, car_h - 60); ctx.fill()
    ctx.rectangle(cx - 3, car_y + 110, 6, 24); ctx.fill()
    ctx.rectangle(cx - 12, car_y + 119, 24, 6); ctx.fill(); ctx.restore()
    
    ctx.save(); rounded_rect(ctx, car_x + 6, car_y + 6, car_w - 12, 30, 6); ctx.set_source_rgb(0.18, 0.32, 0.45); ctx.fill()
    rounded_rect(ctx, cx - 18, car_y + 42, 14, 10, 3); ctx.set_source_rgb(0.95, 0.1, 0.1); ctx.fill()
    rounded_rect(ctx, cx + 4, car_y + 42, 14, 10, 3); ctx.set_source_rgb(0.95, 0.1, 0.1); ctx.fill(); ctx.restore()
    
    ctx.save(); ctx.set_source_rgb(1.0, 0.98, 0.85)
    rounded_rect(ctx, car_x + 6, car_y + 3, 11, 7, 2); ctx.fill()
    rounded_rect(ctx, car_x + car_w - 17, car_y + 3, 11, 7, 2); ctx.fill()
    ctx.set_source_rgb(0.9, 0.1, 0.1)
    rounded_rect(ctx, car_x + 5, car_y + car_h - 8, 10, 6, 2); ctx.fill()
    rounded_rect(ctx, car_x + car_w - 15, car_y + car_h - 8, 10, 6, 2); ctx.fill(); ctx.restore()
    surface.write_to_png(os.path.join(OUTPUT_DIR, filename))

def generate_bus(filename='bus.png'):
    w, h = 130, 360
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(surface)
    cx = w / 2
    bus_w, bus_h = 84, 324
    bus_x, bus_y = cx - bus_w / 2, 18
    r = 12
    draw_shadow(ctx, bus_x, bus_y, bus_w, bus_h, r)
    draw_wheels(ctx, bus_x, bus_y, bus_w, bus_h, 46, 46, 9, 30)
    draw_wheels(ctx, bus_x, bus_y, bus_w, bus_h, 46, 82, 9, 30)
    
    ctx.save(); rounded_rect(ctx, bus_x, bus_y, bus_w, bus_h, r)
    b_grad = cairo.LinearGradient(bus_x, bus_y, bus_x + bus_w, bus_y)
    b_grad.add_color_stop_rgb(0.0, 0.88, 0.89, 0.92)
    b_grad.add_color_stop_rgb(0.5, 0.98, 0.99, 1.00)
    b_grad.add_color_stop_rgb(1.0, 0.88, 0.89, 0.92)
    ctx.set_source(b_grad); ctx.fill_preserve()
    ctx.set_source_rgba(0.4, 0.45, 0.5, 0.9); ctx.set_line_width(2); ctx.stroke(); ctx.restore()
    
    ctx.save(); ctx.set_source_rgb(0.05, 0.45, 0.85)
    ctx.rectangle(bus_x + 5, bus_y + 36, 4, bus_h - 60); ctx.fill()
    ctx.rectangle(bus_x + bus_w - 9, bus_y + 36, 4, bus_h - 60); ctx.fill(); ctx.restore()
    
    ctx.save(); rounded_rect(ctx, bus_x + 5, bus_y + 4, bus_w - 10, 36, 6); ctx.set_source_rgb(0.12, 0.25, 0.38); ctx.fill()
    ac_w, ac_h = 44, 78
    rounded_rect(ctx, cx - ac_w/2, bus_y + 110, ac_w, ac_h, 6); ctx.set_source_rgb(0.82, 0.84, 0.88); ctx.fill()
    ctx.set_source_rgb(0.35, 0.38, 0.45)
    for gy in range(int(bus_y + 122), int(bus_y + 176), 8):
        ctx.rectangle(cx - 16, gy, 32, 3); ctx.fill()
    ctx.restore()
    
    ctx.save(); ctx.set_source_rgb(1.0, 0.95, 0.75)
    rounded_rect(ctx, bus_x + 6, bus_y + 2, 10, 6, 2); ctx.fill()
    rounded_rect(ctx, bus_x + bus_w - 16, bus_y + 2, 10, 6, 2); ctx.fill()
    ctx.set_source_rgb(0.95, 0.1, 0.1)
    rounded_rect(ctx, bus_x + 6, bus_y + bus_h - 6, 12, 4, 1); ctx.fill()
    rounded_rect(ctx, bus_x + bus_w - 18, bus_y + bus_h - 6, 12, 4, 1); ctx.fill(); ctx.restore()
    surface.write_to_png(os.path.join(OUTPUT_DIR, filename))

def generate_truck(filename='truck.png'):
    w, h = 130, 330
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(surface)
    cx = w / 2
    truck_w, truck_h = 82, 296
    truck_x, truck_y = cx - truck_w / 2, 18
    draw_shadow(ctx, truck_x, truck_y, truck_w, truck_h, 8)
    draw_wheels(ctx, truck_x, truck_y, truck_w, truck_h, 28, 34, 9, 28)
    draw_wheels(ctx, truck_x, truck_y, truck_w, truck_h, 28, 68, 9, 28)
    
    cab_h = 82
    ctx.save(); rounded_rect(ctx, truck_x, truck_y, truck_w, cab_h, 12)
    c_grad = cairo.LinearGradient(truck_x, truck_y, truck_x + truck_w, truck_y)
    c_grad.add_color_stop_rgb(0.0, 0.08, 0.28, 0.58)
    c_grad.add_color_stop_rgb(0.5, 0.15, 0.45, 0.85)
    c_grad.add_color_stop_rgb(1.0, 0.08, 0.28, 0.58)
    ctx.set_source(c_grad); ctx.fill()
    rounded_rect(ctx, truck_x + 6, truck_y + 5, truck_w - 12, 24, 5); ctx.set_source_rgb(0.18, 0.32, 0.45); ctx.fill()
    rounded_rect(ctx, cx - 26, truck_y + 36, 52, 38, 4); ctx.set_source_rgb(0.12, 0.36, 0.70); ctx.fill(); ctx.restore()
    
    box_y = truck_y + cab_h + 8
    box_h = truck_h - cab_h - 8
    ctx.save(); rounded_rect(ctx, truck_x - 1, box_y, truck_w + 2, box_h, 4)
    ctx.set_source_rgb(0.85, 0.87, 0.90); ctx.fill()
    ctx.set_source_rgba(0.50, 0.55, 0.62, 0.6); ctx.set_line_width(1.5)
    for ny in range(int(box_y + 16), int(box_y + box_h - 12), 16):
        ctx.move_to(truck_x + 4, ny); ctx.line_to(truck_x + truck_w - 4, ny); ctx.stroke()
    ctx.restore()
    
    ctx.save(); ctx.set_source_rgb(1.0, 0.98, 0.85)
    rounded_rect(ctx, truck_x + 5, truck_y + 2, 12, 6, 2); ctx.fill()
    rounded_rect(ctx, truck_x + truck_w - 17, truck_y + 2, 12, 6, 2); ctx.fill()
    ctx.set_source_rgb(0.9, 0.1, 0.1)
    rounded_rect(ctx, truck_x + 4, box_y + box_h - 5, 14, 4, 1); ctx.fill()
    rounded_rect(ctx, truck_x + truck_w - 18, box_y + box_h - 5, 14, 4, 1); ctx.fill(); ctx.restore()
    surface.write_to_png(os.path.join(OUTPUT_DIR, filename))

def generate_truck_long(filename='truck_long.png'):
    w, h = 130, 480
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(surface)
    cx = w / 2
    truck_w, truck_h = 82, 446
    truck_x, truck_y = cx - truck_w / 2, 18
    draw_shadow(ctx, truck_x, truck_y, truck_w, truck_h, 8)
    draw_wheels(ctx, truck_x, truck_y, truck_w, 90, 26, 12, 9, 26)
    draw_wheels(ctx, truck_x, truck_y + 110, truck_w, 336, 12, 14, 9, 28)
    draw_wheels(ctx, truck_x, truck_y + 110, truck_w, 336, 12, 46, 9, 28)
    
    cab_h = 82
    ctx.save(); rounded_rect(ctx, truck_x, truck_y, truck_w, cab_h, 12)
    grad = cairo.LinearGradient(truck_x, truck_y, truck_x + truck_w, truck_y)
    grad.add_color_stop_rgb(0.0, 0.75, 0.15, 0.15)
    grad.add_color_stop_rgb(0.5, 0.95, 0.30, 0.30)
    grad.add_color_stop_rgb(1.0, 0.75, 0.15, 0.15)
    ctx.set_source(grad); ctx.fill_preserve()
    ctx.set_source_rgb(0.4, 0.05, 0.05); ctx.set_line_width(2); ctx.stroke()
    rounded_rect(ctx, truck_x + 6, truck_y + 5, truck_w - 12, 24, 5); ctx.set_source_rgb(0.18, 0.32, 0.45); ctx.fill()
    rounded_rect(ctx, cx - 26, truck_y + 36, 52, 38, 4); ctx.set_source_rgb(0.85, 0.20, 0.20); ctx.fill(); ctx.restore()
    
    ctx.save(); ctx.set_source_rgb(0.15, 0.15, 0.18); ctx.rectangle(cx - 14, truck_y + cab_h, 28, 16); ctx.fill(); ctx.restore()
    
    c_y = truck_y + cab_h + 12
    c_h = truck_h - cab_h - 12
    ctx.save(); rounded_rect(ctx, truck_x - 1, c_y, truck_w + 2, c_h, 4)
    grad2 = cairo.LinearGradient(truck_x, c_y, truck_x + truck_w, c_y)
    grad2.add_color_stop_rgb(0.0, 0.75, 0.78, 0.82)
    grad2.add_color_stop_rgb(0.5, 0.90, 0.92, 0.95)
    grad2.add_color_stop_rgb(1.0, 0.75, 0.78, 0.82)
    ctx.set_source(grad2); ctx.fill_preserve()
    ctx.set_source_rgb(0.45, 0.48, 0.52); ctx.set_line_width(2); ctx.stroke()
    ctx.set_source_rgba(0.40, 0.45, 0.50, 0.6); ctx.set_line_width(1.5)
    for ny in range(int(c_y + 16), int(c_y + c_h - 12), 16):
        ctx.move_to(truck_x + 4, ny); ctx.line_to(truck_x + truck_w - 4, ny); ctx.stroke()
    ctx.restore()
    
    ctx.save(); ctx.set_source_rgb(1.0, 0.98, 0.85)
    rounded_rect(ctx, truck_x + 5, truck_y + 2, 12, 6, 2); ctx.fill()
    rounded_rect(ctx, truck_x + truck_w - 17, truck_y + 2, 12, 6, 2); ctx.fill()
    ctx.set_source_rgb(0.9, 0.1, 0.1)
    rounded_rect(ctx, truck_x + 4, c_y + c_h - 5, 14, 4, 1); ctx.fill()
    rounded_rect(ctx, truck_x + truck_w - 18, c_y + c_h - 5, 14, 4, 1); ctx.fill(); ctx.restore()
    surface.write_to_png(os.path.join(OUTPUT_DIR, filename))

def generate_van(filename='van.png'):
    w, h = 120, 250
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(surface)
    cx = w / 2
    van_w, van_h = 78, 212
    van_x, van_y = cx - van_w / 2, 20
    draw_shadow(ctx, van_x, van_y, van_w, van_h, 14)
    draw_wheels(ctx, van_x, van_y, van_w, van_h, 34, 32, 8, 26)
    ctx.save(); rounded_rect(ctx, van_x, van_y, van_w, van_h, 14)
    v_grad = cairo.LinearGradient(van_x, van_y, van_x + van_w, van_y)
    v_grad.add_color_stop_rgb(0.0, 0.22, 0.25, 0.30)
    v_grad.add_color_stop_rgb(0.5, 0.38, 0.42, 0.48)
    v_grad.add_color_stop_rgb(1.0, 0.22, 0.25, 0.30)
    ctx.set_source(v_grad); ctx.fill()
    rounded_rect(ctx, van_x + 6, van_y + 6, van_w - 12, 30, 6); ctx.set_source_rgb(0.18, 0.32, 0.45); ctx.fill()
    ctx.set_source_rgba(0.18, 0.22, 0.28, 0.5); ctx.set_line_width(1.5)
    for vx in [-18, -6, 6, 18]:
        ctx.move_to(cx + vx, van_y + 50); ctx.line_to(cx + vx, van_y + van_h - 18); ctx.stroke()
    ctx.restore()
    
    ctx.save(); ctx.set_source_rgb(1.0, 0.98, 0.85)
    rounded_rect(ctx, van_x + 6, van_y + 3, 11, 7, 2); ctx.fill()
    rounded_rect(ctx, van_x + van_w - 17, van_y + 3, 11, 7, 2); ctx.fill()
    ctx.set_source_rgb(0.9, 0.1, 0.1)
    rounded_rect(ctx, van_x + 5, van_y + van_h - 8, 10, 6, 2); ctx.fill()
    rounded_rect(ctx, van_x + van_w - 15, van_y + van_h - 8, 10, 6, 2); ctx.fill(); ctx.restore()
    surface.write_to_png(os.path.join(OUTPUT_DIR, filename))

def generate_motorcycle(filename='motorcycle.png'):
    w, h = 60, 140
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(surface)
    cx, moto_y = w / 2, 10
    ctx.save(); ctx.set_source_rgba(0, 0, 0, 0.3); ctx.arc(cx, 70, 14, 0, 2*math.pi); ctx.fill(); ctx.restore()
    ctx.save(); ctx.set_source_rgb(0.1, 0.1, 0.12); rounded_rect(ctx, cx - 4, moto_y + 4, 8, 24, 3); ctx.fill()
    ctx.set_source_rgb(1.0, 1.0, 0.9); ctx.arc(cx, moto_y + 2, 8, 0, 2*math.pi); ctx.fill()
    ctx.set_source_rgb(0.18, 0.18, 0.20); ctx.rectangle(cx - 22, moto_y + 22, 44, 4); ctx.fill()
    rounded_rect(ctx, cx - 9, moto_y + 28, 18, 26, 6); ctx.set_source_rgb(0.15, 0.80, 0.35); ctx.fill()
    rounded_rect(ctx, cx - 18, moto_y + 50, 36, 22, 9); ctx.set_source_rgb(0.14, 0.15, 0.18); ctx.fill()
    ctx.save(); ctx.translate(cx, moto_y + 56); ctx.scale(1.0, 1.25); ctx.arc(0, 0, 10, 0, 2*math.pi)
    ctx.set_source_rgb(0.95, 0.95, 0.98); ctx.fill_preserve()
    ctx.set_source_rgb(0.1, 0.1, 0.1); ctx.set_line_width(1.5); ctx.stroke(); ctx.restore()
    rounded_rect(ctx, cx - 7, moto_y + 74, 14, 30, 4); ctx.set_source_rgb(0.12, 0.12, 0.12); ctx.fill()
    rounded_rect(ctx, cx - 4, moto_y + 98, 8, 26, 3); ctx.set_source_rgb(0.1, 0.1, 0.12); ctx.fill()
    rounded_rect(ctx, cx - 6, moto_y + 120, 12, 4, 1); ctx.set_source_rgb(0.95, 0.08, 0.08); ctx.fill(); ctx.restore()
    surface.write_to_png(os.path.join(OUTPUT_DIR, filename))

if __name__ == '__main__':
    print('Gerando pack completo de assets com Pycairo...')
    generate_car_sedan()
    generate_car_hatch()
    generate_car_sports()
    generate_taxi()
    generate_police()
    generate_ambulance()
    generate_bus()
    generate_truck()
    generate_truck_long()
    generate_van()
    generate_motorcycle()
    print('Pack completo gerado com sucesso!')
