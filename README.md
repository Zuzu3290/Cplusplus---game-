# C - plus-plus schenanigan 
Ehhh lets hype up the game 


# Solar System Simulation

This project contains a simple solar system simulation implemented in **C++** using **OpenCL** for computation.

---

## Requirements

Before compiling and running the program, make sure the following are installed:

- **MinGW (g++ compiler)**
- **OpenCL SDK / Drivers**
- A GPU or CPU that supports **OpenCL**

> **Note:** The program will not compile or run correctly without OpenCL installed on your system.

---

## Compile

Open a terminal in the project directory and compile the program using:

```bash
g++ solar_system.cpp -o solar_system -I/mingw64/include -L/mingw64/lib -lSDL2 -lOpenCL -std=c++17
```
```bash
solar_system.exe
``` 
