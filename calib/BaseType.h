#pragma once

#include <vector>
#include <string>


template<typename T>
using vector2D = std::vector<std::vector<T>>;

template<typename T>
using vector3D = std::vector<std::vector<std::vector<T>>>;

template<typename T>
using vector4D = std::vector<std::vector<std::vector<std::vector<T>>>>;

template<typename T>
using Sp = std::shared_ptr<T>;

