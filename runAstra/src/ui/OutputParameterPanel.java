package ui;

import data.InputParameter;
import data.OutputParameter;
import main.Config;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.util.ArrayList;

public class OutputParameterPanel extends JPanel {
    public JTable table;
    public ArrayList<OutputParameter> parameters = new ArrayList<>();

    public OutputParameterPanel(){
        setPreferredSize(new Dimension(Config.boardWidth*2/3, Config.boardHeight));
        setFocusable(true);
        setName("output panel");
        initializeVariables();
    }

    private void initializeVariables(){
        parameters.add(new OutputParameter("X_pos"));
        parameters.add(new OutputParameter("Y_pos"));
        parameters.add(new OutputParameter("Z_pos"));
        parameters.add(new OutputParameter("average_kin_energy"));
        parameters.add(new OutputParameter("alfa_X"));
        parameters.add(new OutputParameter("alfa_Y"));
        parameters.add(new OutputParameter("Charge"));
        parameters.add(new OutputParameter("sigZ"));
        parameters.add(new OutputParameter("sigX"));
        parameters.add(new OutputParameter("sigY"));
        parameters.add(new OutputParameter("energy_spread"));
        parameters.add(new OutputParameter("emittance_Y"));
        parameters.add(new OutputParameter("emittance_X"));
        parameters.get(7).isZAxis = true;
        table = new JTable();
        DefaultTableModel dtm = new OutputParameterTable(parameters);
        table.setModel(dtm);
        dtm.addTableModelListener(table);
        table.setDefaultRenderer(Object.class, new ParameterTableRenderer());
        table.setDefaultRenderer(Boolean.class, new CheckBoxRenderer());


        JScrollPane scrollPane = new JScrollPane(table);

        SpringLayout layoutMain = new SpringLayout();
        layoutMain.putConstraint(SpringLayout.NORTH, scrollPane, 0, SpringLayout.NORTH, this);
        layoutMain.putConstraint(SpringLayout.SOUTH, scrollPane, 0, SpringLayout.SOUTH, this);
        layoutMain.putConstraint(SpringLayout.WEST, scrollPane, 0, SpringLayout.WEST, this);
        layoutMain.putConstraint(SpringLayout.EAST, scrollPane, 0, SpringLayout.EAST, this);
        setLayout(layoutMain);

        add(scrollPane);
    }
}
