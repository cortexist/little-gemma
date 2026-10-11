#include <stdio.h>
#include "turn-control.h"
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)
int main(void) {
    struct turn_control c = {0};
    unsigned char msg[7] = {3, 0x34, 0x12, 0, 0, 1, 0};
    CHECK(!turn_control_parse(&c, msg, 0x1234));
    CHECK(c.action == TURN_CONTROL_SPEAK && !c.name);
    CHECK(turn_control_parse(&c, msg, 0x1235));
    for (int a=0; a<5; a++) {
        msg[5]=(unsigned char)a;
        CHECK((turn_control_parse(&c,msg,0x1234)==0)==(a>=1 && a<=3));
    }
    msg[5]=2; msg[6]=2; CHECK(turn_control_parse(&c,msg,0x1234));
    msg[6]=1; CHECK(!turn_control_parse(&c,msg,0x1234));
    CHECK(!turn_control_name_done(&c, "Dr. Jo"));
    CHECK(!turn_control_name_done(&c, "nes</speaker_"));
    CHECK(turn_control_name_done(&c, "name>"));
    for (unsigned i=0; i<=strlen("</speaker_name>"); i++) {
        char a[30]; memcpy(a,"</speaker_name>",i); a[i]=0;
        CHECK(!turn_control_parse(&c,msg,0x1234));
        int first=turn_control_name_done(&c,a);
        CHECK(first || turn_control_name_done(&c,"</speaker_name>"+i));
    }
    msg[5]=1; CHECK(!turn_control_parse(&c,msg,0x1234));
    CHECK(!turn_control_name_done(&c,"</speaker_name>Hello!"));
    msg[0]=2; CHECK(turn_control_parse(&c,msg,0x1234));
    puts("turn control: OK"); return 0;
}
