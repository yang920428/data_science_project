#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<thread>

using namespace std;

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
    size_t pos = line.find("TxSoil30cm"); // 查找 "TxSoil30cm" 的位置
    if (pos != string::npos) { // 如果找到了
        return line.substr(pos + 10); // 返回 "TxSoil30cm" 后面的内容（加上 "TxSoil30cm" 的长度）
    } else {
        return ""; // 如果没找到，返回空字符串
    }
}

int main(){
    int count = 0;
    string sta = "467530";
    string sta_name = "ALiShan";
    int y = 1;

    // cout << "input sta_name and sta num : ";
    // cin >> sta_name >> sta;

    // cout << "input y : " ;
    // cin >> y ;

    date begin_day;
    begin_day.yr = 2024;
    begin_day.mon = 4;
    begin_day.day = 12;

    

        ifstream input ; 
        string begin_daystr = to_string(begin_day.yr) + "-" + to_string(begin_day.mon) + "-" + to_string(begin_day.day);
        string path = "D://code_sets//ds_bigproject//data_set//raw_data//ALiShan//467530-2024-04-12.csv";
        input.open(path,ios::in);


        if (input) cout << "opend" << endl;

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
            cout <<    extractAfterTxSoil30cm(line) << endl;
        }

        //change begin_day
        
        input.close();

    return 0;
}