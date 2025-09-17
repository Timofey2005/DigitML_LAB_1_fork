WARNINGS = -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wformat=2 \
 -Winit-self -Wmissing-declarations -Wredundant-decls -Wshadow \
 -Wstrict-overflow=5 -Wswitch-default -Wundef

# Переключаемся на C++17
FLAGS = $(WARNINGS) -std=c++17

# Все исходники
SRC = src/main.cpp src/NN.cpp src/dataset.cpp

sigmoid:
	g++ $(FLAGS) -Ofast $(SRC) -I include -o main

bent_identity:
	g++ $(FLAGS) -DPERS -Ofast $(SRC) -I include -o main

# Тесты: добавляем -pthread и -lgtest_main
test:
	g++ $(FLAGS) -DTESTS -Ofast $(SRC) -I include -o main -lgtest -lgtest_main -pthread

all: sigmoid

debug:
	g++ $(FLAGS) -DDEBUG $(SRC) -I include -o main
	./main
 
 
