TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        layer.cpp \
        main.cpp \
        mlp.cpp \
        neuron.cpp

HEADERS += \
  layer.h \
  mlp.h \
  neuron.h
