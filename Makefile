test:
	mkdir -p logs/mobilefacenet
	venv/bin/python main.py models/mobilefacenet.tflite > logs/mobilefacenet/_mobilefacenet.log

	mkdir -p logs/rafael_student
	venv/bin/python main.py models/rafael_student.tflite > logs/rafael_student/_rafael_student.log

	mkdir -p logs/triplet_dist_student
	venv/bin/python main.py models/triplet_dist_student.tflite > logs/triplet_dist_student/_triplet_dist_student.log

	mkdir -p logs/pair_model
	venv/bin/python main.py models/pair_model.tflite > logs/pair_model/_pair_model.log

	mkdir -p logs/pair_paper_model
	venv/bin/python main.py models/pair_paper_model.tflite > logs/pair_paper_model/_pair_paper_model.log
