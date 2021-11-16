#include "process.h"

Process::Process(string datapath, string filepath){
    data = datapath;
    file = filepath;
}
Process::Process(string filepath){
    file = filepath;
}

void Process::list_dir(const char *path, std::vector<std::string> &files) {
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

tm Process::getDate(string s){
    tm tm1;
    sscanf(s.c_str(),"%4d-%2d-%2d %2d:%2d:%2d", &tm1.tm_year, &tm1.tm_mon,
        &tm1.tm_mday, &tm1.tm_hour, &tm1.tm_min, &tm1.tm_sec);
    return tm1;
}

bool Process::IsMorning(tm taken_time){
    if (taken_time.tm_hour >= MORNING_START_TIME && taken_time.tm_hour <= MORNING_END_TIME)
        return true;
    else return false;
}

bool Process::IsNight(tm taken_time){
    if (taken_time.tm_hour > NIGHT_START_TIME || taken_time.tm_hour < NIGHT_END_TIME)
        return true;
    else return false;
}

void Process::ProcessingAffine(){
    int64 start = cv::getTickCount();
    string input_path = data + "/o/";
    string output_path = data + "/f1/";
    string log_path = data + "/l/";

    std::ifstream filetxtCount(file);
    std::ifstream filetxt(file);
    std::string info;

    int nImage = std::count(std::istreambuf_iterator<char>(filetxtCount),
        std::istreambuf_iterator<char>(), '\n');
    filetxtCount.close();

    int n_error =0;
    Mat morning_reference;
    string morningFileReference;
    Mat night_reference;
    string nightFileReference;
    Mat img_reference;
    string imageFileReference;
    tm pre_taken_time;

    //read folder log(l) to get image_reference
    std::vector<std::string> pre_image_reference;
    list_dir("/media/master/data/timelapse/files/1VBKkRNgF1fxMjeT2X3KyLtZGQM8a8BJL/l", pre_image_reference);
    for (int i=0; i<pre_image_reference.size();i++){
        if (pre_image_reference[i] == "morning_reference.jpg"){
            morning_reference = cv::imread(log_path+"morning_reference.jpg");
            morningFileReference = "morning_reference.jpg";
        }
        else if (pre_image_reference[i] == "night_reference.jpg"){
            night_reference = cv::imread(log_path+"night_reference.jpg");
            nightFileReference = "night_reference.jpg";
        }
    }

    bool success;
    int n, k;
    double process_time;
    double total_time;
    string log;

    std::time_t t = std::time(0);   // get time now
    std::tm* date = std::localtime(&t);
    string logName = "log1-"+to_string(date->tm_mon+1)+"-"+to_string(date->tm_year+1900)+".txt";
    ofstream logfile;
    logfile.open(log_path+logName, ios::app);
    while (std::getline(filetxt, info)){
        try{
            int64 t_start = cv::getTickCount();
            string name, datestring;
            k = info.find('|');
            if (k != -1){
                name = info.substr(0, k);
                datestring = info.substr(k+1);
            }
            string image_path = input_path + name;
            tm taken_time = getDate(datestring);

            Mat img_align = cv::imread(image_path);

            string output_file = output_path + name;
            bool changeMorning = false;
            bool changeNight = false;
            if (IsMorning(taken_time)){
                // check first morning image
                if (morning_reference.rows == 0) {
                    morning_reference = cv::imread(image_path);
                    img_reference = morning_reference;
                    morningFileReference = name;
                    imageFileReference = name;
                    pre_taken_time = taken_time;
                    cv::imwrite(output_file, morning_reference);
                                       
                    cout << nImage << ": " << name << "   First morning image"<<endl;
                    // log
                    log = "success    " + name + "    " + imageFileReference +"\n";
                    logfile << log;
                    
                    nImage -= 1;
                    continue;                    
                }
                else if (pre_taken_time.tm_mday != taken_time.tm_mday || !IsMorning(pre_taken_time)){
                    changeMorning = true;
                    img_reference = morning_reference;
                    imageFileReference = morningFileReference;
                }
            }
            // else if (IsNight(taken_time)){
            //     // check first night image
            //     if (night_reference.rows == 0){
            //         night_reference = cv::imread(image_path);
            //         img_reference = night_reference;
            //         nightFileReference = name;
            //         imageFileReference = name;
            //         pre_taken_time = taken_time;
            //         cv::imwrite(output_file, night_reference);
                                       
            //         cout << nImage << ": " << name << "   First night image"<<endl;
            //         // log
            //         log = "success    " + name + "    " + imageFileReference +"\n";
            //         logfile << log;
                    
            //         nImage -= 1;
            //         continue;
            //     }
            //     else if (taken_time.tm_hour > NIGHT_START_TIME && pre_taken_time.tm_hour <= NIGHT_START_TIME){
            //         changeNight = true;
            //         img_reference = night_reference;
            //         imageFileReference = nightFileReference;
            //     }
            // }

            AffineSurf affineSurf(img_align, img_reference);
            success = affineSurf.warpImage();
            n = affineSurf.getNumberMatches();
            int64 t_end = cv::getTickCount();
            process_time = (t_end - t_start)/cv::getTickFrequency();
            process_time = round(process_time*100)/100;

            if (!success){
                cout << nImage << ": " << name << "     unsuccess      number_match: "<< n <<"    process_time: "<<to_string(process_time)<<endl;
                //log
                log = "unsuccess    "+ name + "    "+ imageFileReference + "    "+ to_string(process_time)+"\n";
                logfile << log;
                
                nImage -= 1;
                n_error ++;
                continue;
            }

            Mat imReg = affineSurf.getRegistration();
            cv::imwrite(output_file, imReg);

            cout << nImage <<": " << name << "     number_match: "<< n <<"    process_time: "<<to_string(process_time)<<endl;
            //log
            log = "success    "+ name + "    "+ imageFileReference + "    "+ to_string(process_time)+"\n";
            logfile << log;
            //store
            if (changeMorning) {
                morning_reference = imReg;
                morningFileReference = name;
            }
            if (changeNight) {
                night_reference = imReg;
                nightFileReference = name;
            }
            img_reference = imReg;
            imageFileReference = name;
            pre_taken_time = taken_time;
            nImage -= 1;
        } catch( Exception e){
            // cout << e << endl;
        }
    }
    int64 end = cv::getTickCount();
    total_time = round((end-start)/cv::getTickFrequency());
    if (morning_reference.rows) cv::imwrite(log_path+"morning_reference.jpg", morning_reference);
    if (night_reference.rows) cv::imwrite(log_path+"night_reference.jpg", night_reference);
    cout << "total_error: "<< to_string(n_error) << endl;
    cout << "total_time: "<< to_string(total_time) << endl;
    logfile << "total_error: "+to_string(n_error)+"\n";
    logfile << "total_time: "+to_string(total_time)+"\n";
    logfile.close();
}

void Process::ProcessingHomography(string view, string folder_name){
    int64 start = cv::getTickCount();

    // create path
    string input_path = data + "/o/";
    string output_path = data + "/f1/";
    string log_path = data + "/l/";
    
    if (folder_name.length()){
        input_path += folder_name + "/";
        output_path += folder_name + "/";
        log_path += folder_name + "/";
    }

    // read file
    std::ifstream filetxtCount(file);
    std::ifstream filetxt(file);
    std::string info;

    // read number Images (Lines)in file
    int nImage = std::count(std::istreambuf_iterator<char>(filetxtCount),
        std::istreambuf_iterator<char>(), '\n');
    filetxtCount.close();

    // create variable reference
    int n_error =0;
    Mat morning_reference;
    string morningFileReference;
    Mat night_reference;
    string nightFileReference;
    Mat img_reference;
    string imageFileReference;
    Mat old_reference;
    string oldFileReference;
    tm pre_taken_time;

    //read folder log(l) to get image_reference
    std::vector<std::string> pre_image_reference;
    list_dir(log_path.c_str(), pre_image_reference);
    for (int i=0; i<pre_image_reference.size();i++){
        if (pre_image_reference[i] == "morning_reference.jpg"){
            morning_reference = cv::imread(log_path+"morning_reference.jpg");
            morningFileReference = "morning_reference.jpg";
        }
        else if (pre_image_reference[i] == "night_reference.jpg"){
            night_reference = cv::imread(log_path+"night_reference.jpg");
            nightFileReference = "night_reference.jpg";
        }
    }

    bool success = false;
    int n, k;
    double process_time;
    double total_time;
    string log;

    // create file log
    std::time_t t = std::time(0);   // get time now
    std::tm* date = std::localtime(&t);
    string logName = "log1-"+to_string(date->tm_mon+1)+"-"+to_string(date->tm_year+1900)+".txt";
    ofstream logfile;
    logfile.open(log_path+logName, ios::app);

    // create json object (if view = json)
    json results;

    while (std::getline(filetxt, info)){
        try{

            int64 t_start = cv::getTickCount();
            string name, datestring;
            k = info.find('|');
            if (k != -1){
                name = info.substr(0, k);
                datestring = info.substr(k+1);
            }
            string image_path = input_path + name;
            tm taken_time = getDate(datestring);

            Mat img_align = cv::imread(image_path);

            string output_file = output_path + name;
            bool changeMorning = false;
            bool changeNight = false;
            if (IsMorning(taken_time)){
                // check first morning image
                if (morning_reference.rows == 0) {
                    morning_reference = cv::imread(image_path);
                    img_reference = morning_reference;
                    morningFileReference = name;
                    imageFileReference = name;
                    pre_taken_time = taken_time;

                    old_reference = morning_reference;
                    oldFileReference = name;

                    cv::imwrite(output_file, morning_reference);
                    if (view == "json"){
                        json obj;
                        obj["name"] = name;
                        obj["image"] = name;
                        obj["success"] = true;
                        obj["shift"] = 0;
                        results["data"].push_back(obj);
                    } else if (view == "live"){
                        cout << nImage << ": " << name << "   First morning image"<<endl;
                    }                   

                    // log
                    log = "success    " + name + "    " + imageFileReference +"\n";
                    logfile << log;
                    
                    nImage -= 1;
                    continue;                    
                }
                else if (pre_taken_time.tm_mday != taken_time.tm_mday || !IsMorning(pre_taken_time)){
                    changeMorning = true;
                    img_reference = morning_reference;
                    imageFileReference = morningFileReference;
                }
            }
            // else if (IsNight(taken_time)){
            //     // check first night image
            //     if (night_reference.rows == 0){
            //         night_reference = cv::imread(image_path);
            //         img_reference = night_reference;
            //         nightFileReference = name;
            //         imageFileReference = name;
            //         pre_taken_time = taken_time;
            //         cv::imwrite(output_file, night_reference);
                                       
            //         cout << nImage << ": " << name << "   First night image"<<endl;
            //         // log
            //         log = "success    " + name + "    " + imageFileReference +"\n";
            //         logfile << log;
                    
            //         nImage -= 1;
            //         continue;
            //     }
            //     else if (taken_time.tm_hour > NIGHT_START_TIME && pre_taken_time.tm_hour <= NIGHT_START_TIME){
            //         changeNight = true;
            //         img_reference = night_reference;
            //         imageFileReference = nightFileReference;
            //     }
            // }

            HomographySurf homographySurf(img_align, img_reference);
            success = homographySurf.warpImage(true);

            if (!success) {
                img_reference = old_reference;
                imageFileReference = oldFileReference;
                homographySurf.setReference(img_reference);
                success = homographySurf.warpImage(true);
            }

            n = homographySurf.getNumberMatches();
            int64 t_end = cv::getTickCount();
            process_time = (t_end - t_start)/cv::getTickFrequency();
            process_time = round(process_time*100)/100;

            if (!success){
                if (view == "json"){
                    json obj;
                    obj["name"] = name;
                    obj["image"] = imageFileReference;
                    obj["success"] = false;
                    obj["shift"] = 0;
                    results["data"].push_back(obj);
                } else if (view == "live"){
                    cout << nImage << ": " << name << "     unsuccess      number_match: "<< n <<"    process_time: "<<to_string(process_time)<<endl;
                }
                //log
                log = "unsuccess    "+ name + "    "+ imageFileReference + "    "+ to_string(process_time)+"\n";
                logfile << log;
                
                nImage -= 1;
                n_error ++;
                continue;
            }

            Mat imReg;
            if (homographySurf.getMove()){
                // int shift = homographySurf.createMoving(output_path, name);
                int shift = homographySurf.getShift();
                changeMorning = true;
                old_reference = img_align;
                oldFileReference = name;
                imReg = img_align;
                cv::imwrite(output_file, imReg);    

                int64 t_end = cv::getTickCount();
                process_time = (t_end - t_start)/cv::getTickFrequency();
                process_time = round(process_time*100)/100;
                if (view == "json"){
                    json obj;
                    obj["name"] = name;
                    obj["image"] = imageFileReference;
                    obj["success"] = true;
                    obj["shift"] = shift;
                    results["data"].push_back(obj);
                } else if(view == "live"){
                    cout << nImage <<": " << name << "   move to this projective and use this image      "<< to_string(process_time)<<std::endl;
                }
                //log
                log = "success   "+ name +"     "+ "move image with     "+to_string(process_time)+"\n";

                homographySurf.setMove(false);
            }
            else{
                imReg = homographySurf.getRegistration();
                cv::imwrite(output_file, imReg);

                if (view == "json"){
                    json obj;
                    obj["name"] = name;
                    obj["image"] = imageFileReference;
                    obj["success"] = true;
                    obj["shift"] = 0;
                    results["data"].push_back(obj);
                } else if(view =="live"){
                    cout << nImage <<": " << name << "     number_match: "<< n <<"    process_time: " << to_string(process_time)<<endl;
                }
                //log
                log = "success    "+ name + "    "+ imageFileReference + "    "+ to_string(process_time)+"\n";
            }
            
            logfile << log;
            //store
            if (changeMorning) {
                morning_reference = imReg;
                morningFileReference = name;
            }
            if (changeNight) {
                night_reference = imReg;
                nightFileReference = name;
            }
            img_reference = imReg;
            imageFileReference = name;
            pre_taken_time = taken_time;
            nImage -= 1;
        } catch( Exception e){
            // cout << e << endl;
        }
    }
    int64 end = cv::getTickCount();
    total_time = round((end-start)/cv::getTickFrequency());
    if (morning_reference.rows) cv::imwrite(log_path+"morning_reference.jpg", morning_reference);
    if (night_reference.rows) cv::imwrite(log_path+"night_reference.jpg", night_reference);

    // log
    logfile << "total_error: "+to_string(n_error)+"\n";
    logfile << "total_time: "+to_string(total_time)+"\n";
    logfile.close();
    if (view == "live"){
        cout << "total_error: "<< to_string(n_error) << endl;
        cout << "total_time: "<< to_string(total_time) << endl;
    }
    if (view == "json"){
        results["total_time"] = total_time;
        results["total_error"] = n_error;
        std::cout<<results.dump() << std::endl;
    }
}

void Process::ProcessingJSON(string view, bool GPU){
    try {
        int64 start = cv::getTickCount();
        std::ifstream data_file(file);
        json jdata = json::parse(data_file);
        string datapath = jdata["path"];

        // create path
        string input_path = datapath + "/o/";
        string output_path = datapath + "/f1/";
        string log_path = datapath + "/l/";

        // read file
        std::string info;

        // read number Images (Lines)in file
        int nImage = jdata["file"].size();

        // create variable reference
        int n_error =0;
        Mat morning_reference;
        string morningFileReference;
        Mat night_reference;
        string nightFileReference;
        Mat img_reference;
        string imageFileReference;
        Mat old_reference;
        string oldFileReference;

        Mat tmp_reference;
        string tmpFileReference;

        tm pre_taken_time;

        string _imageFileReference;

        morningFileReference = jdata.value("origin","");
        if (morningFileReference != "")  morning_reference = cv::imread(input_path+morningFileReference);

        //read folder log(l) to get image_reference
        std::vector<std::string> pre_image_reference;
        list_dir(log_path.c_str(), pre_image_reference);
        for (int i=0; i<pre_image_reference.size();i++){
            if (pre_image_reference[i] == "morning_reference.jpg"){
                morning_reference = cv::imread(log_path+"morning_reference.jpg");
                morningFileReference = "morning_reference.jpg";
	
                old_reference = morning_reference;
                oldFileReference = morningFileReference;
                img_reference = morning_reference;
                imageFileReference = morningFileReference;	
            }
            // else if (pre_image_reference[i] == "night_reference.jpg"){
            //     night_reference = cv::imread(log_path+"night_reference.jpg");
            //     nightFileReference = "night_reference.jpg";
            // }
        }

        bool success = false;
        bool use_old = false;
        int n, k;
        double process_time;
        double total_time;
        string log;
        int nkp1 = 0;
        int nkp2 = 0;

        // create file log
        std::time_t t = std::time(0);   // get time now
        std::tm* date = std::localtime(&t);
        string logName = "log1-"+to_string(date->tm_mon+1)+"-"+to_string(date->tm_year+1900)+".txt";
        ofstream logfile;
        logfile.open(log_path+logName, ios::app);
	
		// create file json
		std::size_t found = file.find_last_of(".");
		string json_log = file.substr(0,found);
		json_log = json_log + ".json";
		ofstream logjson;
		logjson.open(json_log, ios::app);
        // create json object (if view = json)
        json results;

        for(int i=0; i<jdata["file"].size(); i++){
            try{
                int64 t_start = cv::getTickCount();

                nkp1 = 0;
                string name, datestring;
                string info = jdata["file"][i];
                k = info.find('|');
                if (k != -1){
                    name = info.substr(0, k);
                    datestring = info.substr(k+1);
                }
                string image_path = input_path + name;
                tm taken_time = getDate(datestring);

                Mat img_align = cv::imread(image_path);
                
                if ((!IsMorning(taken_time) && morning_reference.empty())||img_align.empty()){
                    // log json
					json obj;
					obj["name"] = name;
					obj["image"] = "";
					obj["success"] = false;
					obj["shift"] = 0;
					obj["time"] = 0;
					results["data"].push_back(obj);
					if (view == "live"){
                        cout << nImage << ": " << name << "     unsuccess      can't read image"<<endl;
                    }
                    //log
                    log = "unsuccess    "+ name + "    "+ "can't read image"+"\n";
                    logfile << log;
                    
                    nImage -= 1;
                    n_error ++;
                    continue;
                }

                string output_file = output_path + name;
                bool changeMorning = false;
                bool changeNight = false;
                if (IsMorning(taken_time)){
                    // check first morning image
                    if (morning_reference.rows == 0) {
                        morning_reference = cv::imread(image_path);
                        img_reference = morning_reference;
                        morningFileReference = name;
                        imageFileReference = name;
                        pre_taken_time = taken_time;

                        old_reference = morning_reference;
                        oldFileReference = name;

                        cv::imwrite(output_file, morning_reference);
						
						// log json
						json obj;
						obj["name"] = name;
						obj["image"] = name;
						obj["success"] = true;
						obj["shift"] = 0;
						obj["time"] = 0;
                        obj["match"] = 0;
                        obj["keypoint"] = 0;
						results["data"].push_back(obj);
						if (view == "live"){
                            cout << nImage << ": " << name << "   First morning image"<<endl;
                        }                   

                        // log
                        log = "success    " + name + "    " + imageFileReference +"\n";
                        logfile << log;
                        
                        nImage -= 1;
                        continue;                    
                    }
                    else if (pre_taken_time.tm_mday != taken_time.tm_mday || !IsMorning(pre_taken_time)){
                        changeMorning = true;
                        tmp_reference = img_reference;
                        tmpFileReference = imageFileReference;

                        img_reference = morning_reference;
                        imageFileReference = morningFileReference;
                    }
                }
                // else if (IsNight(taken_time)){
                //     // check first night image
                //     if (night_reference.rows == 0){
                //         night_reference = cv::imread(image_path);
                //         img_reference = night_reference;
                //         nightFileReference = name;
                //         imageFileReference = name;
                //         pre_taken_time = taken_time;
                //         cv::imwrite(output_file, night_reference);
                                        
                //         cout << nImage << ": " << name << "   First night image"<<endl;
                //         // log
                //         log = "success    " + name + "    " + imageFileReference +"\n";
                //         logfile << log;
                        
                //         nImage -= 1;
                //         continue;
                //     }
                //     else if (taken_time.tm_hour > NIGHT_START_TIME && pre_taken_time.tm_hour <= NIGHT_START_TIME){
                //         changeNight = true;
                //         img_reference = night_reference;
                //         imageFileReference = nightFileReference;
                //     }
                // }

                if (img_align.size() != img_reference.size()) {
                    cv::resize(img_align, img_align, img_reference.size());
                }
		 
                HomographySurf homographySurf(img_align, img_reference);
                success = homographySurf.warpImage(GPU);

                if (!success && changeMorning) {
                    // img_reference = tmp_reference;
                    // imageFileReference = tmpFileReference;
                    homographySurf.setReference(tmp_reference);
                    success = homographySurf.warpImage(GPU);
                    use_old = true;
		            //cout << "unsucess test " << name << endl; 
                }
                
                nkp1 = homographySurf.getNumberKp1();
                nkp2 = homographySurf.getNumberKp2();
                n = homographySurf.getNumberMatches();
                int64 t_end = cv::getTickCount();
                process_time = (t_end - t_start)/cv::getTickFrequency();
                process_time = round(process_time*100)/100;

                if (use_old) {
                    _imageFileReference = tmpFileReference;
                    use_old = false;
                } else _imageFileReference = imageFileReference;

                if (!success || (homographySurf.getMove() && !IsMorning(taken_time)) || (homographySurf.getMove() && nkp1<MIN_KEYPOINT)){
					// log json
					json obj;
					obj["name"] = name;
					obj["image"] = _imageFileReference;
					obj["success"] = false;
					obj["shift"] = 0;
					obj["time"] = process_time;
                    obj["match"] = n;
                    obj["keypoint"] = nkp1;
					results["data"].push_back(obj);
					if (view == "live"){
                        cout << nImage << ": " << name << "     "<< _imageFileReference << "    unsuccess" << "   number_match: "<< n <<"    process_time: "<<to_string(process_time)<<endl;
                    }
                    //log
                    log = "unsuccess    "+ name + "    "+ _imageFileReference + "    "+ to_string(n) +"      "+to_string(nkp1) + "      " +to_string(process_time)+"\n";
                    logfile << log;
                    
                    nImage -= 1;
                    n_error ++;
                    continue;
                }

                Mat imReg;
                if (homographySurf.getMove() && IsMorning(taken_time)){      // co chuyen goc
                    int shift = homographySurf.getShift();
                    changeMorning = true;
                    old_reference = img_align;
                    oldFileReference = name;
                    imReg = img_align;
                    
                    cv::imwrite(output_file, imReg);    
                    // write origin image
                    cv::imwrite(log_path+"morning_reference_"+name, img_reference);

                    int64 t_end = cv::getTickCount();
                    process_time = (t_end - t_start)/cv::getTickFrequency();
                    process_time = round(process_time*100)/100;
					
					// log json
					json obj;
					obj["name"] = name;
					obj["image"] = _imageFileReference;
					obj["success"] = true;
					obj["shift"] = shift;
					obj["time"] = process_time;
                    obj["match"] = n;
                    obj["keypoint"] = nkp1;
					results["data"].push_back(obj);
                    if(view == "live"){
                        cout << nImage <<": " << name << "   move to this projective and use this image      "<< to_string(process_time)<<std::endl;
                    }
                    //log
                    log = "success   "+ name +"     "+ "move image with     "+ _imageFileReference + "      "+to_string(nkp1) + "      "  + to_string(process_time)+"\n";

                    homographySurf.setMove(false);
                }
                else{
                    imReg = homographySurf.getRegistration();
                    cv::imwrite(output_file, imReg);

					// log json
					json obj;
					obj["name"] = name;
					obj["image"] = _imageFileReference;
					obj["success"] = true;
					obj["shift"] = 0;
					obj["time"] = process_time;
                    obj["match"] = n;
                    obj["keypoint"] = nkp1;
					results["data"].push_back(obj);
                    if(view =="live"){
                        cout << nImage <<": " << name << "     "<< _imageFileReference <<"    number_match: "<< n <<"    process_time: "<<to_string(process_time)<<endl;
                    }
                    //log
                    log = "success    "+ name + "    "+ _imageFileReference + "    "+ to_string(n) +"      "+to_string(nkp1) + "      "  + to_string(process_time)+"\n";
                }
                
                logfile << log;
                //store
                if (changeMorning && nkp1 >= MIN_KEYPOINT && nkp1>=nkp2*0.8) {
                    morning_reference = imReg;
                    morningFileReference = name;
                }
                if (changeNight) {
                    night_reference = imReg;
                    nightFileReference = name;
                }
                img_reference = imReg;
                imageFileReference = name;
                pre_taken_time = taken_time;
                nImage -= 1;
            } catch( Exception e){
                std::cout << e.what() << std::endl;
            }
        }
        int64 end = cv::getTickCount();
        total_time = round((end-start)/cv::getTickFrequency());
        if (morning_reference.rows) cv::imwrite(log_path+"morning_reference.jpg", morning_reference);
        if (night_reference.rows) cv::imwrite(log_path+"night_reference.jpg", night_reference);

        // log
        logfile << "total_error: "+to_string(n_error)+"\n";
        logfile << "total_time: "+to_string(total_time)+"\n";
        logfile.close();
		results["total_time"] = total_time;
		results["total_error"] = n_error;
		logjson << results;
		logjson.close();
        if (view == "live"){
            cout << "total_error: "<< to_string(n_error) << endl;
            cout << "total_time: "<< to_string(total_time) << endl;
        }
		
		if (view == "json"){
            std::cout<<results.dump() << std::endl;
        }
    } catch ( Exception e){

    }
}
