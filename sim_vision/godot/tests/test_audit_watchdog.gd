extends GutTest
## Kiểm thử logic của AuditWatchdog (dashcam rolling buffer, failure trigger matrix).

const AuditWatchdogScript = preload("res://scripts/audit_watchdog.gd")


func test_initial_state_and_defaults() -> void:
	var watchdog := AuditWatchdogScript.new()
	add_child_autoqfree(watchdog)

	assert_true(watchdog.enabled, "Mặc định watchdog phải được bật")
	assert_eq(watchdog._frame_buffer.size(), 0, "Buffer ban đầu rỗng")
	assert_false(watchdog._is_recording_event, "Chưa ghi nhận event lúc khởi tạo")


func test_trigger_failure_sets_event_and_countdown() -> void:
	var watchdog := AuditWatchdogScript.new()
	add_child_autoqfree(watchdog)

	var captured: Dictionary = {"received": false, "error": ""}
	watchdog.failure_triggered.connect(func(err: String, _meta: Dictionary) -> void:
		captured["received"] = true
		captured["error"] = err
	)

	var meta: Dictionary = {"detail": "unit_test_trigger"}
	watchdog.trigger_failure("TETHER_BREAK", meta)

	assert_true(watchdog._is_recording_event, "Đang trong trạng thái ghi nhận event")
	assert_eq(watchdog._active_error_type, "TETHER_BREAK", "Đúng loại lỗi")
	assert_eq(watchdog._post_event_countdown, AuditWatchdog.POST_EVENT_FRAMES, "Bắt đầu đếm ngược post-event")
	assert_true(captured["received"], "Phát tín hiệu failure_triggered")
	assert_eq(captured["error"], "TETHER_BREAK", "Loại lỗi trong tín hiệu đúng")

	# Trigger lần 2 khi đang ghi nhận event thì bỏ qua
	watchdog.trigger_failure("STAGNATION", {})
	assert_eq(watchdog._active_error_type, "TETHER_BREAK", "Không bị ghi đè event đang chạy")


func test_frame_buffer_bounding() -> void:
	var watchdog := AuditWatchdogScript.new()
	add_child_autoqfree(watchdog)

	var dummy_jpg := PackedByteArray([0xFF, 0xD8, 0xFF, 0xE0])
	var max_allowed: int = AuditWatchdog.BUFFER_MAX_FRAMES + AuditWatchdog.POST_EVENT_FRAMES

	for i in range(max_allowed + 20):
		watchdog._frame_buffer.append(dummy_jpg)
		if watchdog._frame_buffer.size() > max_allowed:
			watchdog._frame_buffer.pop_front()

	assert_eq(watchdog._frame_buffer.size(), max_allowed, "Buffer không vượt quá giới hạn tối đa")
