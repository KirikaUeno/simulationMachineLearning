package data;

public class SimulationParameter {
    private final String name;
    private double value;
    private double lowerValue;
    private double upperValue;
    private double stepQuantity;
    public boolean isStatic = true;

    public SimulationParameter(String name, double value) {
        this.name = name;
        this.value = value;
        lowerValue = value;
        upperValue = value;
        stepQuantity = 1;
    }

    public String toString() {
        return name + "|" + value;
    }

    public void setValue(int column, String sVal) {
        System.out.println(column);
        try {
            double localValue = Double.parseDouble(sVal);
            switch (column){
                case 1:
                    lowerValue=localValue;
                    System.out.println("case 1");
                    break;
                case 2:
                    upperValue=localValue;
                    System.out.println("case 2");
                    break;
                case 3:
                    stepQuantity=localValue;
                    System.out.println("case 3");
                    break;
                default:
                    break;
            }
        } catch (NumberFormatException e) {
            e.printStackTrace();
        }
    }

    public void setJustValue(String sVal){
        try {
            value = Double.parseDouble(sVal);
        } catch (NumberFormatException e) {
            e.printStackTrace();
        }
    }

    public double getValue() {
        return value;
    }
    public double getLowerValue() {
        return lowerValue;
    }
    public double getUpperValue() {
        return upperValue;
    }
    public double getStep() {
        return stepQuantity;
    }
    public String getName() {
        return name;
    }
}
