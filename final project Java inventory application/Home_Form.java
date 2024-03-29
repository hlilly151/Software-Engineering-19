package java_inventory_application;

import javax.swing.ImageIcon;
import javax.swing.JFrame;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author 1BestCsharp
 */
public class Home_Form extends javax.swing.JFrame {

    /**
     * Creates new form Home_Form
     */
    
    public Home_Form() {
        initComponents();

        ImageIcon imgThisImg = new ImageIcon("src\\images\\my_store.png");

        jLabel_BackgroundImage.setIcon(imgThisImg);

      
        
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jLabel_BackgroundImage = new javax.swing.JLabel();
        jMenuBar1 = new javax.swing.JMenuBar();
        jMenu_PRODUCT_ = new javax.swing.JMenu();
        jMenu_CATEGORY_ = new javax.swing.JMenu();
        jMenu_CUSTOMER_ = new javax.swing.JMenu();
        jMenu_ORDER_ = new javax.swing.JMenu();
        jMenu5_USER_ = new javax.swing.JMenu();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        jMenu_PRODUCT_.setBackground(new java.awt.Color(249, 105, 14));
        jMenu_PRODUCT_.setForeground(new java.awt.Color(255, 255, 255));
        jMenu_PRODUCT_.setText("  PRODUCT  |");
        jMenu_PRODUCT_.setCursor(new java.awt.Cursor(java.awt.Cursor.HAND_CURSOR));
        jMenu_PRODUCT_.setFont(new java.awt.Font("Segoe UI", 1, 20)); // NOI18N
        jMenu_PRODUCT_.setOpaque(true);
        jMenu_PRODUCT_.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                jMenu_PRODUCT_MouseClicked(evt);
            }
        });
        jMenu_PRODUCT_.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jMenu_PRODUCT_ActionPerformed(evt);
            }
        });
        jMenuBar1.add(jMenu_PRODUCT_);

        jMenu_CATEGORY_.setBackground(new java.awt.Color(249, 105, 14));
        jMenu_CATEGORY_.setForeground(new java.awt.Color(255, 255, 255));
        jMenu_CATEGORY_.setText("  CATEGORY  |");
        jMenu_CATEGORY_.setCursor(new java.awt.Cursor(java.awt.Cursor.HAND_CURSOR));
        jMenu_CATEGORY_.setFont(new java.awt.Font("Segoe UI", 1, 20)); // NOI18N
        jMenu_CATEGORY_.setOpaque(true);
        jMenu_CATEGORY_.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                jMenu_CATEGORY_MouseClicked(evt);
            }
        });
        jMenuBar1.add(jMenu_CATEGORY_);

        jMenu_CUSTOMER_.setBackground(new java.awt.Color(249, 105, 14));
        jMenu_CUSTOMER_.setForeground(new java.awt.Color(255, 255, 255));
        jMenu_CUSTOMER_.setText("  CUSTOMER  |");
        jMenu_CUSTOMER_.setCursor(new java.awt.Cursor(java.awt.Cursor.HAND_CURSOR));
        jMenu_CUSTOMER_.setFont(new java.awt.Font("Segoe UI", 1, 20)); // NOI18N
        jMenu_CUSTOMER_.setOpaque(true);
        jMenu_CUSTOMER_.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                jMenu_CUSTOMER_MouseClicked(evt);
            }
        });
        jMenuBar1.add(jMenu_CUSTOMER_);

        jMenu_ORDER_.setBackground(new java.awt.Color(249, 105, 14));
        jMenu_ORDER_.setForeground(new java.awt.Color(255, 255, 255));
        jMenu_ORDER_.setText("  ORDER  |");
        jMenu_ORDER_.setCursor(new java.awt.Cursor(java.awt.Cursor.HAND_CURSOR));
        jMenu_ORDER_.setFont(new java.awt.Font("Segoe UI", 1, 20)); // NOI18N
        jMenu_ORDER_.setOpaque(true);
        jMenu_ORDER_.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                jMenu_ORDER_MouseClicked(evt);
            }
        });
        jMenuBar1.add(jMenu_ORDER_);

        jMenu5_USER_.setBackground(new java.awt.Color(249, 105, 14));
        jMenu5_USER_.setForeground(new java.awt.Color(255, 255, 255));
        jMenu5_USER_.setText("  USER  ");
        jMenu5_USER_.setCursor(new java.awt.Cursor(java.awt.Cursor.HAND_CURSOR));
        jMenu5_USER_.setFont(new java.awt.Font("Segoe UI", 1, 20)); // NOI18N
        jMenu5_USER_.setOpaque(true);
        jMenu5_USER_.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                jMenu5_USER_MouseClicked(evt);
            }
        });
        jMenuBar1.add(jMenu5_USER_);

        setJMenuBar(jMenuBar1);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jLabel_BackgroundImage, javax.swing.GroupLayout.DEFAULT_SIZE, 1200, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jLabel_BackgroundImage, javax.swing.GroupLayout.PREFERRED_SIZE, 750, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(0, 0, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jMenu_PRODUCT_ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jMenu_PRODUCT_ActionPerformed
        
    }//GEN-LAST:event_jMenu_PRODUCT_ActionPerformed

    // open the product form
    private void jMenu_PRODUCT_MouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_jMenu_PRODUCT_MouseClicked
        
                MANAGE_PRODUCTS_FORM productForm = new MANAGE_PRODUCTS_FORM();
                productForm.pack();
                productForm.setVisible(true);
                productForm.setLocationRelativeTo(null);
                productForm.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        
    }//GEN-LAST:event_jMenu_PRODUCT_MouseClicked

    // open the category form
    private void jMenu_CATEGORY_MouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_jMenu_CATEGORY_MouseClicked
        
        MANAGE_CATEGORIES_FORM categoryForm = new MANAGE_CATEGORIES_FORM();
                categoryForm.pack();
                categoryForm.setVisible(true);
                categoryForm.setLocationRelativeTo(null);
                categoryForm.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        
    }//GEN-LAST:event_jMenu_CATEGORY_MouseClicked

    // open the customer form
    private void jMenu_CUSTOMER_MouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_jMenu_CUSTOMER_MouseClicked
        
        MANAGE_CUSTOMERS_FORM customerForm = new MANAGE_CUSTOMERS_FORM();
                customerForm.pack();
                customerForm.setVisible(true);
                customerForm.setLocationRelativeTo(null);
                customerForm.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        
    }//GEN-LAST:event_jMenu_CUSTOMER_MouseClicked

    // open the order form
    private void jMenu_ORDER_MouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_jMenu_ORDER_MouseClicked
        
        MANAGE_ORDERS_FORM orderForm = new MANAGE_ORDERS_FORM();
                orderForm.pack();
                orderForm.setVisible(true);
                orderForm.setLocationRelativeTo(null);
                orderForm.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        
    }//GEN-LAST:event_jMenu_ORDER_MouseClicked

    // open the user form
    private void jMenu5_USER_MouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_jMenu5_USER_MouseClicked
        
        MANAGE_USERS_FORM userForm = new MANAGE_USERS_FORM();
                userForm.pack();
                userForm.setVisible(true);
                userForm.setLocationRelativeTo(null);
                userForm.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        
    }//GEN-LAST:event_jMenu5_USER_MouseClicked

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(Home_Form.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(Home_Form.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(Home_Form.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(Home_Form.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new Home_Form().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    public javax.swing.JLabel jLabel_BackgroundImage;
    public javax.swing.JMenu jMenu5_USER_;
    private javax.swing.JMenuBar jMenuBar1;
    private javax.swing.JMenu jMenu_CATEGORY_;
    private javax.swing.JMenu jMenu_CUSTOMER_;
    private javax.swing.JMenu jMenu_ORDER_;
    private javax.swing.JMenu jMenu_PRODUCT_;
    // End of variables declaration//GEN-END:variables
}
