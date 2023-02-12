// print a function which given 3 points, (x,y) 
// returns the equation of the circle passing through the 3 points
function circle(x1,x2,x3,y1,y2,y3) {    
    var a = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2);
    var b = (x1*x1 + y1*y1)*(y3-y2) + (x2*x2 + y2*y2)*(y1-y3) + (x3*x3 + y3*y3)*(y2-y1);
    var c = (x1*x1 + y1*y1)*(x2-x3) + (x2*x2 + y2*y2)*(x3-x1) + (x3*x3 + y3*y3)*(x1-x2);
    var d = (x1*x1 + y1*y1)*(x3*y2-x2*y3) + (x2*x2 + y2*y2)*(x1*y3-x3*y1) + (x3*x3 + y3*y3)*(x2*y1-x1*y2);
    var x = -b/(2*a);
    var y = -c/(2*a);
    var r = Math.sqrt((b*b+c*c-4*a*d)/(4*a*a));
    return "x^2 + y^2 + " + (-2*x) + "x + " + (-2*y) + "y + " + (x*x + y*y - r*r) + " = 0";
}
// write a main function to test the above function
int main() {
    cout << circle(0,1,2,0,1,2) << endl;
    return 0;
}

