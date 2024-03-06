.PHONY: all clean

all: main

main: main.py
    chmod +x main.py

clean:
    rm -f main