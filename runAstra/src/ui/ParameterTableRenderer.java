package ui;

import javax.swing.*;
import javax.swing.table.DefaultTableCellRenderer;
import java.awt.*;

public class ParameterTableRenderer extends DefaultTableCellRenderer {

    public ParameterTableRenderer() {
        super();
    }

    @Override
    public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int column) {
        Component cell = super.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, column);

        Color c = Color.LIGHT_GRAY;
        if (isSelected) {
            cell.setBackground(c.darker());
        } else {
            cell.setBackground(c);
        }

        return cell;
    }
}
