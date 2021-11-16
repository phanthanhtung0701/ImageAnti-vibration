#include <iostream>
#include <dirent.h>
#include <sys/types.h>
#include <string>
#include <vector>

using namespace std;

void list_dir(const char *path, std::vector<std::string> &files) {
   struct dirent *entry;
   DIR *dir = opendir(path);
   
   if (dir == NULL) {
      return;
   }
   while ((entry = readdir(dir)) != NULL) {
   //cout << entry->d_name << endl;
   files.push_back(entry->d_name);
   }
   closedir(dir);
}
int main() {
    std::vector<std::string> files;
    list_dir("/media/master/data/timelapse/files/1H0l9kl-YBgNgRtGZBb04w2z9dbaS61Vk/l/", files);
    cout<<(files.size())<<endl;
    //cout << files[0];
    if (files[0] == "file.txt") cout<<"aa"<<endl;
    for (int i = 0;i<files.size();i++){
        std::cout<<files[i] <<std::endl;
    }
}