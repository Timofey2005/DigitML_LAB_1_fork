:WARNINGS = -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wformat=2\
 -Winit-self -Wmissing-declarations -Wredundant-decls -Wshadow\
 -Wstrict-overflow=5 -Wswitch-default -Wundef

FLAGS = $(WARNINGS) -std=c++20

SRC = src/main.cpp

sigmoid: 
	g++ $(FLAGS) -Ofast $(SRC) -I include -o main

bent_identity:
	g++ $(FLAGS) -DPERS -Ofast $(SRC) -I include -o main


leaky_relu:
	g++ $(FLAGS) -DLEAKY_RELU -Ofast $(SRC) -I include -o main  # Leaky ReLU



test:
	g++ $(FLAGS) -DTESTS -Ofast $(SRC) -I include -o main -lgtest

all: sigmoid #Потом поменяю на leaky_relu

debug:
	g++ $(FLAGS) -DDEBUG $(SRC) -o main
	./main
