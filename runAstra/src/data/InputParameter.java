package data;

public class InputParameter {
    public String name;
    public double value;

    public InputParameter(String name){
        this.name = name;
        value = 0;
    }

    public String toString(){
        return ""+name+"|"+value;
    }
}
