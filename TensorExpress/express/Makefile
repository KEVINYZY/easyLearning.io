INC = -I./ -I./generated -I../install/include \
	  -I/usr/local/include -I/usr/include 
DEF = 
LINK = -L../install/lib -lATen -lrpc

AR = ar
CC = g++
C = gcc
CFLAGS = -g -std=c++11 -fPIC -Wall $(DEF) $(INC)
LDFLAGS = -L/usr/local/lib $(LINK)
OBJS_DIR = ./obj

APP_SRCS =
include build.mk
APP_SRCS_ = ${APP_SRCS:%.c=$(OBJS_DIR)/%.o}
APP_SRCS__ = ${APP_SRCS_:%.cc=$(OBJS_DIR)/%.o}
APP_OBJS = ${APP_SRCS__:%.cpp=$(OBJS_DIR)/%.o}

TARGET = express

$(OBJS_DIR)/%.o : %.c
	@mkdir -p $(@D)
	$(C) $(CFLAGS) -c -o $@ $<

$(OBJS_DIR)/%.o : %.cc
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJS_DIR)/%.o : %.cpp
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $<

$(TARGET): $(APP_OBJS)
	$(CC) -o $@ $(APP_OBJS) $(LDFLAGS) 

.PHONY: clean all
all: $(TARGET)
clean:
	rm -f $(TARGET)
	rm -rf $(OBJS_DIR)
