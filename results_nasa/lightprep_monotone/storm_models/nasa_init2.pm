dtmc
const int H = 10;

module health_model
    s : [0..2] init 2;
    t : [0..H] init 0;

    [] s=0 & t<H -> 0.9934434828 : (s'=0) & (t'=t+1) + 0.0065565172 : (s'=1) & (t'=t+1) + 0.0000000000 : (s'=2) & (t'=t+1);
    [] s=1 & t<H -> 0.0000000000 : (s'=0) & (t'=t+1) + 0.9768089054 : (s'=1) & (t'=t+1) + 0.0231910946 : (s'=2) & (t'=t+1);
    [] s=2 & t<H -> 0.0000000000 : (s'=0) & (t'=t+1) + 0.0000000000 : (s'=1) & (t'=t+1) + 1.0000000000 : (s'=2) & (t'=t+1);
    [] t=H -> 1.0 : (s'=s) & (t'=t);
endmodule

label "critical" = s=2;

// Example property to check in Storm:
// P=? [ F<=H "critical" ]
