package data;

public class OutputParameter {
    public boolean estimate = true;
    public String name;
    public double estimatedValue;
    public boolean isZAxis = false;

    public OutputParameter(String name){
        this.name = name;
        estimatedValue = 0;
    }
}
