package ui;

import data.SimulationParameter;

import javax.swing.table.DefaultTableModel;
import java.text.DecimalFormat;
import java.util.List;

public class StaticTable extends DefaultTableModel {

    private final DecimalFormat df = new DecimalFormat("0.000");
    private final DecimalFormat dfSmall = new DecimalFormat("0.000E0");

    public List<SimulationParameter> auList;

    public StaticTable(List<SimulationParameter> _auList) {
        this.auList = _auList;
    }

    public int getRowCount() {
        if (auList == null) {
            return 0;
        }
        return auList.size();
    }

    public int getColumnCount() {
        return 2;
    }

    public String getColumnName(int columnIndex) {
        if (columnIndex == 0) {
            return "parameter";
        } else if (columnIndex == 1) {
            return "Value";
        }
        return null;
    }

    public Class<?> getColumnClass(int columnIndex) {
        return Object.class;
    }

    public boolean isCellEditable(int rowIndex, int columnIndex) {
        return columnIndex == 1;
    }

    public Object getValueAt(int rowIndex, int columnIndex) {
        SimulationParameter au = auList.get(rowIndex);

        if (columnIndex == 0) {
            return au.getName();
        } else if (columnIndex == 1) {
            return au.getValue();
        }
        return "";
    }

    public void setValueAt(Object obj, int rowIndex, int columnIndex) {
        SimulationParameter au = auList.get(rowIndex);

        if (columnIndex == 1) {
            au.setJustValue((String)obj);
        }
    }
}
