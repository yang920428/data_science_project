#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<thread>
#include<vector>

using namespace std;

vector<vector<float>> daily_data;

typedef struct date
{
    int yr , mon , day;
}date;

bool is_leap_year(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

bool is_big_mon(int mon){
    return (mon ==1 || mon == 3 || mon == 5 || mon == 7 || mon == 8 || mon == 10 || mon == 12);
}

date getYesterday(date day){
    int yr = day.yr;
    int mon = day.mon;
    int d = day.day;

    if (d > 1) {
        date yes;
        yes.yr = yr;
        yes.mon = mon;
        yes.day = d - 1;
        return yes;
    }
    else if (mon > 1 && is_big_mon(mon-1)){
        date yes;
        yes.yr = yr;
        yes.mon = mon-1;
        yes.day = 31;
        return yes;
    }
    else if (mon > 1 && !is_big_mon(mon-1) && mon != 3){
        date yes;
        yes.yr = yr;
        yes.mon = mon-1;
        yes.day = 30;
        return yes;
    }
    else if (mon == 3 && is_leap_year(yr)){
        date yes;
        yes.yr = yr;
        yes.mon = mon-1;
        yes.day = 29;
        return yes;
    }
    else if (mon == 3 && !is_leap_year(yr)){
        date yes;
        yes.yr = yr;
        yes.mon = mon-1;
        yes.day = 28;
        return yes;
    }
    else{
        date yes;
        yes.yr = yr-1;
        yes.mon = 12;
        yes.day = 31;
        return yes;
    }
}

string extractAfterTxSoil30cm(const string& line) {
    //size_t pos = line.find("TxSoil30cm"); // 查找 "TxSoil30cm" 的位置
    size_t pos = line.find("Cloud Amount");
    if (pos != string::npos) { // 如果找到了
        return line.substr(pos + 10+2); // 返回 "TxSoil30cm" 后面的内容（加上 "TxSoil30cm" 的长度）
    } else {
        return ""; // 如果没找到，返回空字符串
    }
}

string fills(int n){
    if (n/10 ==0){
        return "0"+to_string(n);
    }
    else{
        return to_string(n);
    }
}

string processCSVLine(const string& line) {
    stringstream ss(line);
    string field;
    string result;

    while (getline(ss, field, ',')) {
        size_t start = 0;
        size_t end = field.size();

        if (field.front() == '"' && field.back() == '"') {
            start = 1;
            end = field.size() - 1;
        }

        result += field.substr(start, end - start) + " ";
    }

    // 去掉所有的双引号
    size_t pos = 0;
    while ((pos = result.find("\"", pos)) != string::npos) {
        result.erase(pos, 1);
    }
    
    return result;
}

void parseData(const string& data) {
    vector<vector<float>> parsedData;

    istringstream iss(data);
    string line;

    while (getline(iss, line)) {
        //istringstream lineStream(line);
        string token;
        vector<float> rowData;

        string lines = line;

        while (lines.find(" ")!= -1)
        {
            token = lines.substr(0 , lines.find(" "));
            try {
                rowData.push_back(stof(token));
            }
            catch(const std::invalid_argument& e){
                if (token == "T"){
                    rowData.push_back(0.2);
                }
                else{
                    rowData.push_back(-1);
                }
            }
            lines = lines.substr(lines.find(" ")+1 , lines.size());
        }
        token = lines;
        try {
                rowData.push_back(stof(token));
            }
            catch(const std::invalid_argument& e){
                if (token == "T"){
                    rowData.push_back(0.2);
                }
                else{
                    rowData.push_back(-1);
                }
            }
        
        // while (getline(lineStream, token, ' ')) {
        //     try {
        //         rowData.push_back(stof(token));
        //     }
        //     catch(const std::invalid_argument& e){
        //         if (token == "T"){
        //             rowData.push_back(0.2);
        //         }
        //         else{
        //             rowData.push_back(-1);
        //         }
        //     }
        //     // if (token != "--" && token != "T" && token.size()>0) {
        //     //     // Convert each token to float if it's not "--"
        //     //     rowData.push_back(stof(token));
        //     // } else {
        //     //     if (token == "T"){
        //     //         rowData.push_back(0.2);
        //     //     }
        //     //     else if (token.size() == 0){

        //     //     }
        //     //     else{
        //     //         rowData.push_back(-1.0);
        //     //     }
        //     //     // If token is "--", push back a placeholder value
        //     //      // Or any other placeholder value you prefer
        //     // }
        // }

        // Add the row data to the parsedData vector
        parsedData.push_back(rowData);
        //cout << "pushed row" << endl;
    }

    cout << "function value returned" << endl;

    daily_data = parsedData;
}

int main(){
    int count = 0;
    string sta;
    string sta_name;
    int y = 0;

    cout << "input sta_name and sta num : ";
    cin >> sta_name >> sta;

    cout << "input y : " ;
    cin >> y ;

    date begin_day;
    begin_day.yr = 2024;
    begin_day.mon = 4;
    begin_day.day = 13;

    ofstream txtfile ;
    string path_out = "D://code_sets//ds_bigproject//data_set//txtdata//"+sta_name+"_"+sta+".txt";
    txtfile.open(path_out , ios::out);
    
    while (count < y*365){
        ifstream input ; 
        string begin_daystr = to_string(begin_day.yr) + "-" + (fills(begin_day.mon)) + "-" + (fills(begin_day.day));
        //string path = "D://code_sets//ds_bigproject//data_set//raw_data//ALiShan//467530-2024-04-12.csv";
        string path = "D://code_sets//ds_bigproject//data_set//raw_data//"+sta_name+"//"+sta+"-"+begin_daystr+".csv";
        input.open(path,ios::in);


        //if (input) cout << "opend" << endl;

        string line;
        while (getline(input,line , '\r'))
        {   
            
            int line_counter = 0;   
            //cout << line << endl;
            // stringstream ss(line);
            // string data;
            // while (getline(ss, data, ',')) { // 使用逗号作为分隔符
            //     // cout << "round " << line_counter << endl;
            //     // if (line_counter >=2) cout << data << endl;
            //     // line_counter ++;
            // }


            // string op =      extractAfterTxSoil30cm(line);
            // if (op.size() >=50 ) {
            //     op = processCSVLine(extractAfterTxSoil30cm(line));
            //     cout <<    processCSVLine(extractAfterTxSoil30cm(line)) << endl;
                
            // }


            //cout <<"string read" << endl;
            string op =  extractAfterTxSoil30cm(line);
            cout << op.size() << endl;
            if (op.size() >=50 ) {
                //cout << "in the if " << endl;
                op = processCSVLine(extractAfterTxSoil30cm(line));
                //cout <<    processCSVLine(extractAfterTxSoil30cm(line)) << endl;
                parseData(op);
                cout << "get return value" << endl;
                // if (daily_data.empty()) cout << "empty" << endl;

                cout << "the size is : "<<daily_data.size() << endl;
                //0 2 3 4 9 10 12
                for (int i=1 ; i< daily_data.size() ; i++){
                    vector<float> hour = daily_data[i];
                    //cout << hour[0] << endl;
                    //if (hour.empty()) cout <<"hour empty" << endl;
                    txtfile << hour[0+1] <<" " << hour[2+1] << " " << hour[3+1] << " " << hour[4+1] << " " << hour[9+1] << " " << hour[10+1] << " " << hour[12+1] << endl;
                }
                cout << "write terminated" << endl;
                daily_data.clear();
                // cout << daily_data.size() << endl;
                // cout << daily_data[0][0] << endl;
                //cout << op << endl;
            }
            else{
                //cout <<"i love you yiru" << endl;
            }

        }

        //change begin_day
        begin_day = getYesterday(begin_day);
        count++;
        input.close();
    }
    txtfile.close();
    cout << "programe terminated" << endl;
    return 0;
}