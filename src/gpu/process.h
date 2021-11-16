#ifndef PROCESS_H_
#define PROCESS_H_

#include "affinesurf.h"
#include "homographysurf.h"
#include <ctime>
#include <string>
#include <fstream>
#include <stdio.h>
#include <dirent.h>
#include <sys/types.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class Process{
    private:
        string data;
        string file;
    public:
        Process(string datapath, string filepath);
        Process(string filepath);
        void list_dir(const char *path, std::vector<std::string> &files);
        tm getDate(string s);
        bool IsMorning(tm taken_time);
        bool IsNight(tm taken_time);

        void ProcessingAffine();
        void ProcessingHomography(string view, string folder_name="");
        void ProcessingJSON(string view, bool GPU = false);
};

#endif