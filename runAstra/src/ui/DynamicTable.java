package ui;

import data.SimulationParameter;

import javax.swing.table.DefaultTableModel;
import java.util.List;

public class DynamicTable extends DefaultTableModel {
    public List<SimulationParameter> auList;

    public DynamicTable(List<SimulationParameter> _auList) {
        this.auList = _auList;
    }

    public int getRowCount() {
        if (auList == null) {
            return 0;
        }
        return auList.size();
    }

    public int getColumnCount() {
        return 4;
    }

    public String getColumnName(int columnIndex) {
        switch (columnIndex){
            case 0:
                return "parameter";
            case 1:
                return "lower value";
            case 2:
                return "upper value";
            case 3:
                return "step (for stepped)";
            default:
                return null;
        }
    }

    public Class<?> getColumnClass(int columnIndex) {
        return Object.class;
    }

    public boolean isCellEditable(int rowIndex, int columnIndex) {
        return columnIndex != 0;
    }

    public Object getValueAt(int rowIndex, int columnIndex) {
        SimulationParameter au = auList.get(rowIndex);

        switch (columnIndex){
            case 0:
                return au.getName();
            case 1:
                return au.getLowerValue();
            case 2:
                return au.getUpperValue();
            case 3:
                return au.getStep();
            default:
                return null;
        }
    }

    public void setValueAt(Object obj, int rowIndex, int columnIndex) {
        SimulationParameter au = auList.get(rowIndex);

        au.setValue(columnIndex,(String)obj);
    }
}
