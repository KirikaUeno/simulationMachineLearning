package data;

public class OutputParameter {
    public boolean estimate = true;
    public String name;
    public double estimatedValue;

    public OutputParameter(String name){
        this.name = name;
        estimatedValue = 0;
    }
}
