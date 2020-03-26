DEBUG:= -D DEBUG

TARGET    = waveglow_tts
SRC_DIR   = waveglow/src
SRC_DIR_SYS   = sys/src
SRC_DIR_COMMON   = common/src

OBJ_DIR   = waveglow/obj
OBJ_DIR_SYS   = sys/obj
OBJ_DIR_COMMON   = common/obj

INCLUDES:=-Iwaveglow/header/ -Icommon/header -Isys/header  -I/opt/rocm-3.1.0/include/ -I/opt/rocm/hipdnn/include -I/opt/rocm/hiprand/include -I/opt/rocm-3.1.0/rocrand/include #-I/usr/local/cuda/include

NVCC:=hipcc
LDFLAGS:=   -lcudnn -lcublas -lcurand 
NVCCFLAGS:=  -std=c++11   #-amdgpu-target=gfx906 #--compiler-bindir=g++-4.9 #--ptxas-options=-v -arch=sm_70
CUDNN_PATH:= /usr/local/cuda/
LIBS:= -L $(CUDNN_PATH)/lib64 -L/usr/local/lib  -L/opt/rocm-3.1.0/lib/  -L/opt/rocm/lib   -L/opt/rocm/hipdnn/lib -L/opt/rocm-3.1.0/rocrand/lib -L/opt/rocm/hipdnn/lib
#-L/opt/rocm-3.1.0/hsa/lib/ 



CU_FILES  = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(CU_FILES)))

CU_FILES_SYS  = $(wildcard $(SRC_DIR_SYS)/*.cpp)
OBJS_SYS = $(patsubst %.cpp,$(OBJ_DIR_SYS)/%.o,$(notdir $(CU_FILES_SYS)))

CU_FILES_COMMON  = $(wildcard $(SRC_DIR_COMMON)/*.cpp)
OBJS_COMMON = $(patsubst %.cpp,$(OBJ_DIR_COMMON)/%.o,$(notdir $(CU_FILES_COMMON)))


$(TARGET) :	dirmake $(OBJS_COMMON)	$(OBJS_SYS)	$(OBJS)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $(LIBS) -o $@ $(OBJS_COMMON) $(OBJS_SYS) $(OBJS)


$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $< $(DEBUG)

$(OBJ_DIR_SYS)/%.o : $(SRC_DIR_SYS)/%.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $< $(DEBUG)

$(OBJ_DIR_COMMON)/%.o : $(SRC_DIR_COMMON)/%.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $< $(DEBUG)


dirmake:
	@mkdir -p $(OBJ_DIR_COMMON)
	@mkdir -p $(OBJ_DIR_SYS)
	@mkdir -p $(OBJ_DIR)

.PHONY :	clean
clean :	
	rm -f $(TARGET)
	rm -f $(OBJ_DIR_COMMON)/*.o
	rm -f $(OBJ_DIR_SYS)/*.o
	rm -f $(OBJ_DIR)/*.o

rebuild:	clean build
