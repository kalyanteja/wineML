.requirements_installed:
	pip install --upgrade pip; \
	pip install wheel; \
	pip install -r requirements.txt --use-deprecated=legacy-resolver; \
	touch .requirements_installed

install: .requirements_installed
	echo "Installation Complete"

clean:
	deactivate; \
	rm -rf .requirements_installed

train:
	python drug.py